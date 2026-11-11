from __future__ import annotations
import asyncio, json, os, re, shlex, signal, shutil, sqlite3, subprocess, sys, time, uuid
from collections import deque
from pathlib import Path
from typing import Any
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('AGENT_PRO_HOME', str(Path.home() / '.agent-prototyping-app')))
DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / 'agent-pro.db'
DEFAULT_SYSTEM = "I am a helpful, precise, and reliable AI agent. I follow the user's instructions, use available skills and tools when useful, explain important actions, and clearly distinguish facts from assumptions."
app = FastAPI(title='Agent Prototyping App', version='1.1.6')
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:8750','http://localhost:8750'], allow_methods=['*'], allow_headers=['*'])
app.mount('/static', StaticFiles(directory=ROOT/'app'/'static'), name='static')


def parse_mcp_args(value):
    if value is None or value == '': return []
    if isinstance(value, list): return value
    if isinstance(value, str):
        try: return shlex.split(value)
        except ValueError as e: raise HTTPException(400, f'Invalid command-line arguments: {e}')
    raise HTTPException(400, 'Command-line arguments must be plain text')

def parse_mcp_env(value):
    if value is None or value == '': return {}
    if isinstance(value, dict): return value
    if isinstance(value, str):
        result={}
        for line_no, line in enumerate(value.splitlines(), 1):
            line=line.strip()
            if not line or line.startswith('#'): continue
            if '=' not in line: raise HTTPException(400, f'Environment variable line {line_no} must use KEY=value format')
            key,val=line.split('=',1); key=key.strip()
            if not key or not key.replace('_','').isalnum(): raise HTTPException(400, f'Invalid environment variable name on line {line_no}')
            result[key]=val.strip()
        return result
    raise HTTPException(400, 'Environment variables must use KEY=value lines')

def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

def init_db():
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,name TEXT NOT NULL,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,project_id TEXT NOT NULL,title TEXT NOT NULL,created REAL NOT NULL,updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,details TEXT,created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS skills(id TEXT PRIMARY KEY,name TEXT NOT NULL,kind TEXT NOT NULL,path TEXT NOT NULL,enabled INTEGER NOT NULL DEFAULT 1,command TEXT,args TEXT,env TEXT,example_usage TEXT NOT NULL DEFAULT '',created REAL NOT NULL,transport TEXT NOT NULL DEFAULT 'auto',endpoint TEXT NOT NULL DEFAULT '');
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        ''')
        if not c.execute('SELECT 1 FROM projects LIMIT 1').fetchone():
            c.execute('INSERT INTO projects VALUES(?,?,?)',('default','Unnamed project',time.time()))
        cols={r['name'] for r in c.execute('PRAGMA table_info(skills)').fetchall()}
        if 'example_usage' not in cols: c.execute("ALTER TABLE skills ADD COLUMN example_usage TEXT NOT NULL DEFAULT ''")
        if 'transport' not in cols: c.execute("ALTER TABLE skills ADD COLUMN transport TEXT NOT NULL DEFAULT 'auto'")
        if 'endpoint' not in cols: c.execute("ALTER TABLE skills ADD COLUMN endpoint TEXT NOT NULL DEFAULT ''")
        defaults={'provider':'ollama','model':'llama3.2','base_url':'http://127.0.0.1:11434','api_key':'','system_prompt':DEFAULT_SYSTEM,'temperature':'0.7','top_p':'0.9','top_k':'40','num_ctx':'8192','repeat_penalty':'1.1','max_tokens':'2048','reasoning_enabled':'false','theme':'dark'}
        for k,v in defaults.items(): c.execute('INSERT OR IGNORE INTO settings VALUES(?,?)',(k,v))
        # One-time migration to the requested darker graphite default; later user theme choices persist.
        if not c.execute("SELECT 1 FROM settings WHERE key='theme_migrated_v2'").fetchone():
            c.execute("UPDATE settings SET value='dark' WHERE key='theme'")
            c.execute("INSERT OR REPLACE INTO settings(key,value) VALUES('theme_migrated_v2','1')")
init_db()

class Obj(BaseModel):
    data: dict[str,Any] = Field(default_factory=dict)

class MCPProcess:
    def __init__(self, skill): self.skill=skill; self.proc=None; self.tools=[]; self.lock=asyncio.Lock(); self.stderr_task=None; self.logs=deque(maxlen=600); self.transport='stdio'; self.endpoint=None; self.http_session_id=None; self.starting=False; self.startup_error=None
    async def start(self):
        async with self.lock:
            # Do not spawn a second server when an HTTP process is alive but its
            # first MCP handshake failed. Reinitialize that same process instead.
            if self.proc and self.proc.returncode is None:
                if self.transport == 'http' and self.endpoint and self.startup_error:
                    last_error = None
                    for attempt in range(1, 31):
                        if not self.proc or self.proc.returncode is not None:
                            raise RuntimeError('MCP process exited while reconnecting to its HTTP endpoint')
                        try:
                            await self.rpc('initialize', {'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'agent-prototyping-app','version':'1.1.1'}}, timeout=8)
                            await self.notify('notifications/initialized', {})
                            result = await self.rpc('tools/list', {})
                            self.tools = result.get('tools', [])
                            self.startup_error = None
                            self.logs.append(f'[{time.strftime("%H:%M:%S")}] MCP reinitialized successfully over HTTP; discovered {len(self.tools)} tool(s)')
                            return
                        except Exception as exc:
                            last_error = exc
                            self.logs.append(f'[{time.strftime("%H:%M:%S")}] HTTP reconnect attempt {attempt}/30 failed: {type(exc).__name__}: {exc}')
                            await asyncio.sleep(1)
                    self.startup_error = f'{type(last_error).__name__}: {last_error}' if last_error else 'HTTP initialization timed out'
                    raise RuntimeError(f'MCP process is alive, but HTTP initialization failed: {self.startup_error}')
                return
            # A process can exit between UI refreshes; discard its stale state.
            if self.proc and self.proc.returncode is not None:
                self.proc = None
                self.endpoint = None
                self.http_session_id = None
                self.tools = []
                self.transport = 'stdio'
                self.startup_error = None
            command_text=str(self.skill.get('command') or '').strip()
            extra_args=json.loads(self.skill.get('args') or '[]')
            if not command_text: raise RuntimeError('MCP launch command is not configured')
            try: command_parts=shlex.split(command_text)
            except ValueError as exc: raise RuntimeError(f'Invalid launch command: {exc}')
            if not command_parts: raise RuntimeError('MCP launch command is empty')
            command=command_parts[0]
            args=command_parts[1:]+extra_args
            self.logs.append(f'[{time.strftime("%H:%M:%S")}] Starting: {command_text} {' '.join(extra_args)}'.strip())
            env=os.environ.copy(); env.update(json.loads(self.skill.get('env') or '{}'))
            cwd=self.skill.get('path') or None
            if cwd and not Path(cwd).expanduser().is_dir(): raise RuntimeError(f'MCP working directory does not exist or is not a directory: {cwd}')
            workdir=Path(cwd).expanduser().resolve() if cwd else None
            # Python MCP projects get an isolated environment of their own. The app .venv
            # runs this application; it is not a dependency bucket for every MCP project.
            python_names={'python','python3','python3.9','python3.10','python3.11','python3.12','python3.13'}
            if Path(command).name in python_names:
                python_exe=await self._prepare_python_environment(workdir)
                command=str(python_exe)
                env['VIRTUAL_ENV']=str(python_exe.parent.parent)
                env['PATH']=str(python_exe.parent)+os.pathsep+env.get('PATH','')
            else:
                app_venv_bin=ROOT/'.venv'/'bin'
                if app_venv_bin.is_dir():
                    env['VIRTUAL_ENV']=str(ROOT/'.venv')
                    env['PATH']=str(app_venv_bin)+os.pathsep+env.get('PATH','')
            resolved=command if Path(command).is_file() else shutil.which(command, path=env.get('PATH'))
            if not resolved and not Path(command).expanduser().exists(): raise RuntimeError(f'Launch executable not found: {command}. Check the command and ensure it is installed.')
            self.proc=await asyncio.create_subprocess_exec(resolved or command,*args,cwd=str(workdir) if workdir else None,env=env,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            self.logs.append(f'[{time.strftime("%H:%M:%S")}] Process started (PID {self.proc.pid}); initializing MCP protocol')
            self.stderr_task=asyncio.create_task(self._capture_stderr(self.proc))
            self._id=0; self.startup_error=None
            try:
                # FastMCP HTTP servers announce their Streamable HTTP endpoint on stderr.
                # Detect that before attempting stdio, otherwise a healthy HTTP server is
                # incorrectly terminated after waiting for JSON-RPC on its stdout pipe.
                configured_transport=str(self.skill.get('transport') or 'auto').lower()
                configured_endpoint=str(self.skill.get('endpoint') or '').strip()
                endpoint=None
                if configured_transport == 'http':
                    if not configured_endpoint:
                        raise RuntimeError('Streamable HTTP transport requires an MCP endpoint URL in the server settings')
                    endpoint=configured_endpoint.replace('://0.0.0.0:', '://127.0.0.1:').replace('://[::]:', '://127.0.0.1:')
                    self.transport='http'; self.endpoint=endpoint
                    self.logs.append(f'[{time.strftime("%H:%M:%S")}] Using configured Streamable HTTP MCP endpoint: {endpoint}')
                elif configured_transport == 'auto':
                    endpoint=await self._detect_http_endpoint(timeout=12)
                    if endpoint:
                        self.transport='http'; self.endpoint=endpoint
                        self.logs.append(f'[{time.strftime("%H:%M:%S")}] Detected Streamable HTTP MCP endpoint: {endpoint}')
                else:
                    self.transport='stdio'; self.endpoint=None
                if self.transport == 'http':
                    result=None
                    last_error=None
                    # FastMCP/Uvicorn can announce readiness just before the route is accepting
                    # requests. Retry the MCP handshake rather than misclassifying it as stdio.
                    for attempt in range(1,9):
                        try:
                            await self.rpc('initialize',{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'agent-prototyping-app','version':'1.1.1'}},timeout=8)
                            await self.notify('notifications/initialized',{})
                            result=await self.rpc('tools/list',{})
                            break
                        except Exception as exc:
                            last_error=exc
                            self.logs.append(f'[{time.strftime("%H:%M:%S")}] HTTP MCP handshake attempt {attempt}/8 failed: {type(exc).__name__}: {exc}')
                            await asyncio.sleep(1)
                    if result is None:
                        raise RuntimeError(f'HTTP MCP endpoint was detected but did not complete initialization: {last_error}')
                else:
                    self.transport='stdio'; self.endpoint=None
                    await self.rpc('initialize',{'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'agent-prototyping-app','version':'1.1.1'}},timeout=20)
                    await self.notify('notifications/initialized',{})
                    result=await self.rpc('tools/list',{})
                self.tools=result.get('tools',[])
                self.logs.append(f'[{time.strftime("%H:%M:%S")}] MCP initialized successfully over {self.transport}; discovered {len(self.tools)} tool(s)')
            except Exception as exc:
                self.logs.append(f'[{time.strftime("%H:%M:%S")}] STARTUP ERROR: {type(exc).__name__}: {exc}')
                self.startup_error=f'{type(exc).__name__}: {exc}'
                # Keep a live HTTP MCP process available for diagnosis/retry. A protocol
                # handshake failure must not terminate an otherwise healthy HTTP service.
                if self.transport=='http' and self.proc and self.proc.returncode is None:
                    self.logs.append(f'[{time.strftime("%H:%M:%S")}] HTTP process remains running; tool discovery can be retried from Server details')
                elif self.proc and self.proc.returncode is None:
                    self.proc.terminate()
                    try: await asyncio.wait_for(self.proc.wait(),3)
                    except asyncio.TimeoutError: self.proc.kill(); await self.proc.wait()
                    self.proc=None
                elif self.proc:
                    self.logs.append(f'[{time.strftime("%H:%M:%S")}] MCP process exited during initialization (exit code: {self.proc.returncode})')
                self.tools=[]
                raise
    async def _prepare_python_environment(self, workdir: Path | None) -> Path:
        env_root=DATA/'mcp-venvs'/str(self.skill['id'])
        python_exe=env_root/'bin'/'python'
        pip_exe=env_root/'bin'/'pip'
        if not python_exe.exists():
            env_root.parent.mkdir(parents=True,exist_ok=True)
            self.logs.append(f'[{time.strftime("%H:%M:%S")}] Creating MCP virtual environment: {env_root}')
            proc=await asyncio.create_subprocess_exec(sys.executable,'-m','venv',str(env_root),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
            try:
                output,_=await asyncio.wait_for(proc.communicate(),timeout=180)
            except asyncio.TimeoutError:
                proc.kill(); await proc.wait(); raise RuntimeError('Timed out while creating the MCP virtual environment')
            for line in output.decode(errors='replace').splitlines(): self.logs.append(f'[{time.strftime("%H:%M:%S")}] [venv] {line}')
            if proc.returncode:
                raise RuntimeError(f'Could not create MCP virtual environment (exit code {proc.returncode})')
        if not workdir:
            self.logs.append(f'[{time.strftime("%H:%M:%S")}] No MCP working directory configured; using isolated environment without project dependency installation')
            return python_exe
        requirements=workdir/'requirements.txt'
        pyproject=workdir/'pyproject.toml'
        setup_py=workdir/'setup.py'
        source_files=list(workdir.glob('*.py'))
        source_text='\n'.join(f.read_text(errors='ignore') for f in source_files[:80])
        needs_fastmcp=('from fastmcp' in source_text or 'import fastmcp' in source_text)
        marker=env_root/'.dependencies-installed'
        manifests=[]
        if requirements.is_file(): manifests.append(requirements)
        elif pyproject.is_file() or setup_py.is_file(): manifests.append(pyproject if pyproject.is_file() else setup_py)
        if needs_fastmcp:
            declared=''
            for f in manifests:
                try: declared += f.read_text(errors='ignore').lower()
                except Exception: pass
            if 'fastmcp' not in declared:
                manifests.append('fastmcp')
        if not manifests:
            self.logs.append(f'[{time.strftime("%H:%M:%S")}] No requirements.txt, pyproject.toml or setup.py found; no project packages installed')
            return python_exe
        stamp='|'.join(f'{f}:{f.stat().st_mtime_ns}:{f.stat().st_size}' if isinstance(f,Path) else str(f) for f in manifests)
        source_stamp='|'.join(f'{f.name}:{f.stat().st_mtime_ns}:{f.stat().st_size}' for f in source_files[:80])
        stamp += '|' + source_stamp
        if marker.exists() and marker.read_text(errors='ignore')==stamp:
            self.logs.append(f'[{time.strftime("%H:%M:%S")}] Reusing MCP virtual environment; dependencies unchanged: {env_root}')
            return python_exe
        install_commands=[]
        for item in manifests:
            if isinstance(item,Path) and item.name=='requirements.txt':
                install_commands.append([str(pip_exe),'install','-r',str(item)])
            elif isinstance(item,Path):
                install_commands.append([str(pip_exe),'install','-e',str(workdir)])
            else:
                install_commands.append([str(pip_exe),'install',str(item)])
        for install_args in install_commands:
            self.logs.append(f'[{time.strftime("%H:%M:%S")}] Installing MCP project dependencies into {env_root}: {" ".join(install_args[1:])}')
            proc=await asyncio.create_subprocess_exec(*install_args,cwd=str(workdir),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
            try:
                while True:
                    line=await asyncio.wait_for(proc.stdout.readline(),timeout=900)
                    if not line: break
                    self.logs.append(f'[{time.strftime("%H:%M:%S")}] [pip] {line.decode(errors="replace").rstrip()}')
                code=await asyncio.wait_for(proc.wait(),timeout=30)
            except asyncio.TimeoutError:
                proc.kill(); await proc.wait(); raise RuntimeError('Timed out installing MCP project dependencies (15-minute limit)')
            if code:
                raise RuntimeError(f'MCP dependency installation failed (pip exit code {code}); inspect the [pip] lines in Server logs')
        marker.write_text(stamp)
        self.logs.append(f'[{time.strftime("%H:%M:%S")}] MCP dependencies installed successfully')
        return python_exe
        return python_exe

    async def _capture_stderr(self, proc):
        try:
            while True:
                line=await proc.stderr.readline()
                if not line: break
                self.logs.append(f'[{time.strftime("%H:%M:%S")}] {line.decode(errors='replace').rstrip()}')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logs.append(f'[{time.strftime("%H:%M:%S")}] Log reader error: {exc}')
    async def _read_message(self, timeout=20):
        line=await asyncio.wait_for(self.proc.stdout.readline(),timeout=timeout)
        if not line:
            code=self.proc.returncode if self.proc else 'unknown'
            raise RuntimeError(f'MCP server closed its protocol output (exit code: {code}). Check Server logs for details.')
        try: return json.loads(line.decode())
        except Exception as exc:
            self.logs.append(f'[{time.strftime("%H:%M:%S")}] Invalid MCP stdout (expected JSON-RPC): {line.decode(errors='replace').rstrip()}')
            raise RuntimeError(f'MCP server wrote non-JSON data to stdout: {exc}')
    async def _detect_http_endpoint(self, timeout=12):
        pattern=re.compile(r'https?://(?:0\.0\.0\.0|127\.0\.0\.1|localhost|\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+):\d+(?:/[^\s\x1b]*)?')
        deadline=asyncio.get_running_loop().time()+timeout
        while asyncio.get_running_loop().time()<deadline:
            if not self.proc or self.proc.returncode is not None: return None
            candidates=[]; announced_http=False
            for line in self.logs:
                low=line.lower()
                if "transport 'http'" in low or 'transport=\"http\"' in low: announced_http=True
                for match in pattern.finditer(line):
                    candidates.append(match.group(0).rstrip('),.;'))
            if candidates:
                # Prefer /mcp; if FastMCP only prints Uvicorn's root URL, infer its
                # standard Streamable HTTP route once the HTTP transport is announced.
                url=next((u for u in reversed(candidates) if '/mcp' in u), None)
                if not url and announced_http:
                    url=candidates[-1].rstrip('/')+'/mcp'
                if url:
                    return url.replace('://0.0.0.0:', '://127.0.0.1:').replace('://[::]:', '://127.0.0.1:')
            await asyncio.sleep(.15)
        return None
    async def _http_rpc(self, method, params, notification=False):
        if not self.endpoint: raise RuntimeError('MCP HTTP endpoint is not configured')
        self._id+=1
        payload={'jsonrpc':'2.0','method':method,'params':params}
        if not notification: payload['id']=self._id
        headers={'Accept':'application/json, text/event-stream','Content-Type':'application/json','MCP-Protocol-Version':'2024-11-05'}
        if self.http_session_id: headers['Mcp-Session-Id']=self.http_session_id
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0), trust_env=False) as client:
            response=await client.post(self.endpoint,json=payload,headers=headers)
        if response.status_code==404 and self.http_session_id:
            self.http_session_id=None
        response.raise_for_status()
        sid=response.headers.get('mcp-session-id')
        if sid: self.http_session_id=sid
        if notification or response.status_code==202 or not response.content: return {}
        content=response.text
        if 'text/event-stream' in response.headers.get('content-type',''):
            data_lines=[line[5:].strip() for line in content.splitlines() if line.startswith('data:')]
            content='\n'.join(data_lines)
        for line in content.splitlines():
            line=line.strip()
            if not line or not line.startswith('{'): continue
            msg=json.loads(line)
            if 'error' in msg: raise RuntimeError(msg['error'].get('message','MCP request failed'))
            if 'result' in msg: return msg['result']
        raise RuntimeError(f'MCP HTTP endpoint returned no JSON-RPC result (HTTP {response.status_code})')
    async def rpc(self, method, params, timeout=20):
        if self.transport=='http': return await self._http_rpc(method,params)
        self._id+=1; ident=self._id
        payload={'jsonrpc':'2.0','id':ident,'method':method,'params':params}
        self.proc.stdin.write((json.dumps(payload)+'\n').encode()); await self.proc.stdin.drain()
        while True:
            msg=await self._read_message(timeout=timeout)
            if msg.get('id')==ident:
                if 'error' in msg: raise RuntimeError(msg['error'].get('message','MCP request failed'))
                return msg.get('result',{})
    async def notify(self,method,params):
        if self.transport=='http': return await self._http_rpc(method,params,notification=True)
        self.proc.stdin.write((json.dumps({'jsonrpc':'2.0','method':method,'params':params})+'\n').encode()); await self.proc.stdin.drain()
    async def call(self,name,args):
        if not self.proc or self.proc.returncode is not None: await self.start()
        return await self.rpc('tools/call',{'name':name,'arguments':args})
    async def stop(self):
        async with self.lock:
            if self.proc and self.proc.returncode is None:
                self.proc.terminate()
                try: await asyncio.wait_for(self.proc.wait(),5)
                except asyncio.TimeoutError: self.proc.kill(); await self.proc.wait()
            old_proc=self.proc
            self.proc=None
            self.tools=[]; self.endpoint=None; self.http_session_id=None; self.transport='stdio'; self.starting=False; self.startup_error=None
            if old_proc: self.logs.append(f'[{time.strftime("%H:%M:%S")}] Process stopped (exit code: {old_proc.returncode})')

mcp: dict[str,MCPProcess]={}
model_process: subprocess.Popen | None = None

def rows(query,args=()):
    with db() as c: return [dict(x) for x in c.execute(query,args).fetchall()]
def setting_map(): return {x['key']:x['value'] for x in rows('SELECT * FROM settings')}
def get_skill(sid):
    result=rows('SELECT * FROM skills WHERE id=?',(sid,)); return result[0] if result else None

def tool_catalog(enabled_only=True):
    out=[]
    for s in rows('SELECT * FROM skills ORDER BY name'):
        if s['kind']!='mcp' or (enabled_only and not s['enabled']): continue
        proc=mcp.get(s['id']); alive=bool(proc and proc.proc and proc.proc.returncode is None)
        for t in (proc.tools if alive else []):
            out.append({'type':'function','function':{'name':f"{s['id']}__{t['name']}",'description':f"[{s['name']}] {t.get('description','')}",'parameters':t.get('inputSchema',{'type':'object','properties':{}})}})
    return out

@app.get('/',response_class=HTMLResponse)
async def home(): return (ROOT/'app'/'templates'/'index.html').read_text()
@app.get('/api/health')
async def health(): return {'ok':True,'version':'1.1.6'}
@app.get('/api/model/status')
async def model_status():
    cfg=setting_map(); provider=cfg.get('provider','ollama')
    if provider!='ollama': return {'provider':provider,'model_running':False,'models':[]}
    running,models=await ollama_http_status()
    return {'provider':provider,'model_running':running,'models':models,'model_owned':bool(model_process and model_process.poll() is None)}

@app.get('/api/state')
async def state():
    projects=rows('SELECT * FROM projects ORDER BY created')
    sessions=rows('SELECT * FROM sessions ORDER BY updated DESC')
    skills=rows('SELECT id,name,kind,path,enabled,command,args,env,example_usage,transport,endpoint FROM skills ORDER BY created')
    for s in skills:
        p=mcp.get(s['id']); s['running']=bool(p and p.proc and p.proc.returncode is None); s['starting']=bool(p and p.starting); s['error']=p.startup_error if p else None; s['tools']=p.tools if s['running'] else []
    cfg=setting_map(); provider=cfg.get('provider','ollama'); running=False; models=[]
    if provider=='ollama':
        try:
            running,models=await ollama_http_status()
        except Exception: pass
    return {'projects':projects,'sessions':sessions,'skills':skills,'settings':cfg,'model_running':running,'model_owned':bool(model_process and model_process.poll() is None),'models':models}

@app.post('/api/projects')
async def add_project(o:Obj):
    name=str(o.data.get('name') or 'New project').strip()
    pid=str(uuid.uuid4())
    with db() as c: c.execute('INSERT INTO projects VALUES(?,?,?)',(pid,name,time.time()))
    return {'id':pid,'name':name}
@app.patch('/api/projects/{pid}')
async def rename_project(pid:str,o:Obj):
    with db() as c: c.execute('UPDATE projects SET name=? WHERE id=?',(str(o.data.get('name') or 'Unnamed project'),pid))
    return {'ok':True}
@app.delete('/api/projects/{pid}')
async def delete_project(pid:str):
    if pid=='default': raise HTTPException(400,'The default project cannot be deleted')
    with db() as c:
        c.execute('DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE project_id=?)',(pid,)); c.execute('DELETE FROM sessions WHERE project_id=?',(pid,)); c.execute('DELETE FROM projects WHERE id=?',(pid,))
    return {'ok':True}
@app.post('/api/sessions')
async def add_session(o:Obj):
    sid=str(uuid.uuid4()); pid=str(o.data.get('project_id') or 'default'); now=time.time()
    if not rows('SELECT id FROM projects WHERE id=?',(pid,)): raise HTTPException(404,'Project not found. Refresh the workspace and try again.')
    with db() as c: c.execute('INSERT INTO sessions VALUES(?,?,?,?,?)',(sid,pid,'New chat',now,now))
    return {'id':sid,'title':'New chat','project_id':pid}
@app.patch('/api/sessions/{sid}')
async def rename_session(sid:str,o:Obj):
    with db() as c: c.execute('UPDATE sessions SET title=?,updated=? WHERE id=?',(str(o.data.get('title') or 'New chat'),time.time(),sid))
    return {'ok':True}
@app.delete('/api/sessions/{sid}')
async def delete_session(sid:str):
    with db() as c: c.execute('DELETE FROM messages WHERE session_id=?',(sid,)); c.execute('DELETE FROM sessions WHERE id=?',(sid,))
    return {'ok':True}
@app.get('/api/sessions/{sid}/messages')
async def get_messages(sid:str): return rows('SELECT * FROM messages WHERE session_id=? ORDER BY created',(sid,))

async def start_mcp_background(sid: str):
    skill=get_skill(sid)
    if not skill: return
    proc=mcp.setdefault(sid,MCPProcess(skill)); proc.skill=skill; proc.starting=True
    try:
        await proc.start()
        proc.logs.append(f'[{time.strftime("%H:%M:%S")}] Background MCP startup completed successfully')
    except Exception as exc:
        proc.logs.append(f'[{time.strftime("%H:%M:%S")}] Background MCP startup failed: {type(exc).__name__}: {exc}')
    finally:
        proc.starting=False

@app.post('/api/skills')
async def add_skill(o:Obj):
    d=o.data; kind=d.get('kind','mcp'); sid=str(uuid.uuid4()); name=str(d.get('name') or 'New skill').strip(); path=str(d.get('path') or '').strip()
    if kind=='markdown' and (not path or not Path(path).expanduser().is_file()): raise HTTPException(400,'Select an existing skills.md file')
    if kind=='mcp' and not str(d.get('command') or '').strip(): raise HTTPException(400,'MCP launch command is required')
    transport=str(d.get('transport') or 'auto').lower()
    if transport not in {'auto','stdio','http'}: raise HTTPException(400,'Transport must be Auto, STDIO, or Streamable HTTP')
    endpoint=str(d.get('endpoint') or '').strip()
    if transport=='http' and not endpoint: raise HTTPException(400,'Enter the Streamable HTTP MCP endpoint URL')
    if endpoint and not endpoint.startswith(('http://','https://')): raise HTTPException(400,'MCP endpoint must start with http:// or https://')
    args=parse_mcp_args(d.get('args', ''))
    env=parse_mcp_env(d.get('env', ''))
    # Reuse an already-running identical MCP configuration instead of launching a duplicate.
    if kind=='mcp' and d.get('enabled',True):
        args_json=json.dumps(args,sort_keys=True); env_json=json.dumps(env,sort_keys=True)
        for existing in rows("SELECT * FROM skills WHERE kind='mcp' AND enabled=1"):
            proc=mcp.get(existing['id'])
            alive=bool(proc and proc.proc and proc.proc.returncode is None)
            if alive and existing.get('path','')==path and (existing.get('command') or '')==str(d.get('command') or '') and (existing.get('transport') or 'auto')==transport and (existing.get('endpoint') or '')==endpoint and json.dumps(json.loads(existing.get('args') or '[]'),sort_keys=True)==args_json and json.dumps(json.loads(existing.get('env') or '{}'),sort_keys=True)==env_json:
                return {'id':existing['id'],'connected':True,'tools':proc.tools,'error':None,'already_running':True}
    with db() as c: c.execute('INSERT INTO skills(id,name,kind,path,enabled,command,args,env,example_usage,created,transport,endpoint) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',(sid,name,kind,path,1 if d.get('enabled',True) else 0,d.get('command',''),json.dumps(args),json.dumps(env),str(d.get('example_usage') or '').strip(),time.time(),transport,endpoint))
    response={'id':sid,'connected':False,'starting':False,'tools':[],'error':None}
    return response
@app.patch('/api/skills/{sid}')
async def update_skill(sid:str,o:Obj):
    s=get_skill(sid)
    if not s: raise HTTPException(404,'Skill not found')
    d=o.data
    if 'transport' in d and str(d['transport']).lower() not in {'auto','stdio','http'}: raise HTTPException(400,'Transport must be Auto, STDIO, or Streamable HTTP')
    if str(d.get('transport',s.get('transport') or 'auto')).lower()=='http' and not str(d.get('endpoint',s.get('endpoint') or '')).strip(): raise HTTPException(400,'Enter the Streamable HTTP MCP endpoint URL')
    if 'endpoint' in d and str(d['endpoint']).strip() and not str(d['endpoint']).startswith(('http://','https://')): raise HTTPException(400,'MCP endpoint must start with http:// or https://')
    launch_changed = False
    for key in ('path','command','transport','endpoint'):
        if key in d and str(d[key] or '') != str(s.get(key) or ''): launch_changed = True
    normalized={}
    for key in ('args','env'):
        if key in d:
            normalized[key]=parse_mcp_args(d[key]) if key=='args' else parse_mcp_env(d[key])
            existing=json.loads(s.get(key) or ('[]' if key=='args' else '{}'))
            if json.dumps(normalized[key],sort_keys=True) != json.dumps(existing,sort_keys=True): launch_changed=True
    was_running=bool(sid in mcp and mcp[sid].proc and mcp[sid].proc.returncode is None)
    if launch_changed and sid in mcp:
        await mcp[sid].stop(); mcp.pop(sid,None)
    with db() as c:
        for k in ('name','path','command','example_usage','transport','endpoint'):
            if k in d: c.execute(f'UPDATE skills SET {k}=? WHERE id=?',(str(d[k]),sid))
        for k in ('args','env'):
            if k in normalized: c.execute(f'UPDATE skills SET {k}=? WHERE id=?',(json.dumps(normalized[k]),sid))
        if 'enabled' in d: c.execute('UPDATE skills SET enabled=? WHERE id=?',(int(bool(d['enabled'])),sid))
    result={'ok':True,'connected':False,'starting':False,'tools':[],'error':None}
    latest=get_skill(sid)
    # The enabled flag controls whether the agent may use this integration.
    # It must never start or stop the underlying MCP process. Only explicit
    # Start/Stop actions control the server lifecycle.
    if launch_changed and latest['kind']=='mcp' and was_running:
        proc=MCPProcess(latest); mcp[sid]=proc; proc.starting=True
        asyncio.create_task(start_mcp_background(sid)); result['starting']=True
    elif sid in mcp and latest['kind']=='mcp':
        mcp[sid].skill=latest
    return result
@app.delete('/api/skills/{sid}')
async def delete_skill(sid:str):
    if sid in mcp: await mcp[sid].stop(); mcp.pop(sid,None)
    with db() as c: c.execute('DELETE FROM skills WHERE id=?',(sid,))
    return {'ok':True}
@app.post('/api/skills/{sid}/start')
async def start_skill(sid:str):
    s=get_skill(sid)
    if not s or s['kind']!='mcp': raise HTTPException(404,'MCP server not found')
    try:
        p=mcp.setdefault(sid,MCPProcess(s)); p.skill=s
        # Keep all start/restart paths inside MCPProcess.start so they share its
        # lifecycle lock and perform a complete HTTP initialize sequence.
        await p.start()
        return {'ok':True,'tools':p.tools,'running':bool(p.proc and p.proc.returncode is None),'transport':p.transport,'endpoint':p.endpoint}
    except Exception as e: raise HTTPException(500,str(e))
@app.post('/api/skills/{sid}/stop')
async def stop_skill(sid:str):
    if sid in mcp: await mcp[sid].stop()
    return {'ok':True}
@app.get('/api/skills/{sid}/logs')
async def skill_logs(sid:str):
    s=get_skill(sid)
    if not s or s['kind']!='mcp': raise HTTPException(404,'MCP server not found')
    p=mcp.get(sid)
    return {'logs':list(p.logs) if p else ['No process has been started in this application session yet.'], 'running':bool(p and p.proc and p.proc.returncode is None), 'transport':p.transport if p else None, 'endpoint':p.endpoint if p else None}

@app.post('/api/skills/{sid}/refresh')
async def refresh_skill(sid:str):
    if sid not in mcp: raise HTTPException(400,'Start the MCP server first')
    p=mcp[sid]
    try: p.tools=(await p.rpc('tools/list',{})).get('tools',[]); return {'tools':p.tools}
    except Exception as e: raise HTTPException(500,str(e))

@app.put('/api/settings')
async def save_settings(o:Obj):
    allowed={'provider','model','base_url','api_key','system_prompt','temperature','top_p','top_k','num_ctx','repeat_penalty','max_tokens','reasoning_enabled','theme'}
    with db() as c:
        for k,v in o.data.items():
            if k in allowed: c.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(k,str(v)))
    return {'ok':True}
def find_ollama_executable():
    candidate=shutil.which('ollama')
    if candidate: return candidate
    for candidate in ('/opt/homebrew/bin/ollama','/usr/local/bin/ollama','/Applications/Ollama.app/Contents/Resources/ollama'):
        if Path(candidate).is_file(): return candidate
    return None

def find_ollama_serve_pids():
    listing=subprocess.run(['ps','-axo','pid=,command='],capture_output=True,text=True,timeout=4,check=False)
    found=[]
    for line in listing.stdout.splitlines():
        parts=line.strip().split(None,1)
        if len(parts)!=2: continue
        try: pid=int(parts[0])
        except ValueError: continue
        command=parts[1].strip()
        # Match the Ollama daemon, including the macOS app-bundled executable,
        # without matching `ollama list`, `ollama run`, or this app's command.
        # Avoid shell tokenization here: macOS process command lines can contain
        # quoted paths or app-bundle arguments that make shlex.split raise.
        if re.search(r'(?:^|/)ollama(?:\s|$).*\bserve\b',command) and pid!=os.getpid():
            found.append(pid)
    return found

async def ollama_http_status():
    cfg=setting_map(); base=cfg.get('base_url','http://127.0.0.1:11434').strip().rstrip('/')
    if base.endswith('/api'): base=base[:-4]
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            r=await client.get(base+'/api/tags'); r.raise_for_status()
            payload=r.json(); models=sorted({x.get('name') for x in payload.get('models',[]) if x.get('name')})
            return True, models
    except Exception as exc:
        return False, []

@app.post('/api/model/start')
async def start_model():
    global model_process
    cfg=setting_map()
    if cfg.get('provider')!='ollama': raise HTTPException(400,'Ollama controls are available only for the local provider')
    running,_=await ollama_http_status()
    if running: return {'ok':True,'already_running':True}
    if model_process and model_process.poll() is None: return {'ok':True,'already_running':True,'starting':True}
    executable=find_ollama_executable()
    if not executable: raise HTTPException(500,'Ollama executable was not found. Install Ollama or add it to PATH.')
    try:
        model_process=subprocess.Popen([executable,'serve'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        return {'ok':True,'already_running':False,'starting':True,'pid':model_process.pid}
    except Exception as e: raise HTTPException(500,f'Could not start Ollama: {e}')

@app.post('/api/model/stop')
async def stop_model():
    global model_process
    cfg=setting_map()
    if cfg.get('provider')!='ollama': raise HTTPException(400,'Ollama controls are available only for the local provider')
    pids=[]
    if model_process and model_process.poll() is None:
        pids.append(model_process.pid)
    try:
        pids.extend(pid for pid in find_ollama_serve_pids() if pid not in pids)
    except Exception as e:
        # Process inspection can be restricted by macOS privacy/permissions.
        # The HTTP endpoint remains the source of truth; don't turn a healthy
        # externally-managed Ollama daemon into a misleading API error.
        running,_=await ollama_http_status()
        if not running:
            model_process=None
            return {'ok':True,'stopped':True,'already_stopped':True}
        return {'ok':True,'stopped':False,'externally_managed':True,'message':f'Ollama is responding but macOS did not allow process inspection ({e}). Close it from the Ollama menu-bar app to stop it.'}
    stopped=[]
    for pid in pids:
        try:
            os.kill(pid,signal.SIGTERM); stopped.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
    if model_process and model_process.poll() is not None: model_process=None
    # Ollama may be supervised by the macOS menu-bar app. Give it a moment to
    # exit, then report the actual HTTP state rather than trusting the signal.
    running=True
    for _ in range(12):
        await asyncio.sleep(.25)
        running,_=await ollama_http_status()
        if not running: break
    if not running:
        model_process=None
        return {'ok':True,'stopped':True,'pids':stopped}
    if not stopped:
        return {'ok':True,'stopped':False,'externally_managed':True,'message':'Ollama is managed by another process and is still responding.'}
    return {'ok':True,'stopped':False,'externally_managed':True,'pids':stopped,'message':'A stop signal was sent, but Ollama is still responding.'}



def validate_mcp_tool_arguments(full_name: str, args: Any) -> dict:
    """Constrain model-generated arguments to the live MCP tool schema."""
    if not isinstance(args, dict):
        raise ValueError(f"Tool {full_name} expects a JSON object of arguments.")
    try:
        sid, tool_name = full_name.split('__', 1)
    except ValueError:
        raise ValueError(f"Invalid MCP tool identifier: {full_name}")
    proc = mcp.get(sid)
    tool = next((item for item in (proc.tools if proc else []) if item.get('name') == tool_name), None)
    if not tool:
        raise ValueError(f"Tool schema for {tool_name} is unavailable. Refresh the MCP tool list and retry.")
    schema = tool.get('inputSchema') or {}
    properties = schema.get('properties') or {}
    allowed = set(properties)
    cleaned = {key: value for key, value in args.items() if key in allowed}
    unexpected = sorted(set(args) - allowed)

    # Models sometimes confuse a tool identifier with the value returned by
    # that tool. Never send such an identifier as a business parameter.
    for key, value in cleaned.items():
        if isinstance(value, str):
            candidate = value.strip()
            if re.fullmatch(r'[0-9a-fA-F-]{36}__[A-Za-z0-9_.-]+', candidate) or (
                '__' in candidate and any(
                    candidate == f"{other_sid}__{other_tool.get('name')}"
                    for other_sid, other_proc in mcp.items()
                    for other_tool in (other_proc.tools or [])
                )
            ):
                raise ValueError(
                    f"Invalid value for '{key}': '{candidate}' is an MCP tool identifier, "
                    "not the tool's returned data. Call that MCP tool first, then use its actual output."
                )

    required = schema.get('required') or []
    missing = [key for key in required if key not in cleaned or cleaned[key] is None or cleaned[key] == '']
    if missing:
        detail = f"Missing required argument(s): {', '.join(missing)}."
        if unexpected:
            detail += f" Unsupported argument(s) ignored: {', '.join(unexpected)}."
        detail += " Call any prerequisite tool needed to obtain the missing values; do not invent values."
        raise ValueError(detail)
    return cleaned

async def run_mcp_tool(full_name,args):
    sid,name=full_name.split('__',1); p=mcp.get(sid)
    if not p or not p.proc or p.proc.returncode is not None: raise RuntimeError('MCP server is not running')
    return await p.call(name,args)
async def model_chat(messages, tools, cfg):
    provider=cfg.get('provider','ollama'); base=cfg.get('base_url','http://127.0.0.1:11434').rstrip('/'); model=cfg.get('model','llama3.2')
    params={'temperature':float(cfg.get('temperature',.7)),'top_p':float(cfg.get('top_p',.9)),'top_k':int(cfg.get('top_k',40)),'repeat_penalty':float(cfg.get('repeat_penalty',1.1)),'num_ctx':int(cfg.get('num_ctx',8192)),'num_predict':int(cfg.get('max_tokens',2048))}
    async with httpx.AsyncClient(timeout=180) as client:
        if provider=='ollama':
            payload={'model':model,'messages':messages,'stream':False,'options':params}
            if str(cfg.get('reasoning_enabled','false')).lower() in {'1','true','yes','on'}: payload['think']=True
            if tools: payload['tools']=tools
            r=await client.post(base+'/api/chat',json=payload); r.raise_for_status(); return r.json().get('message',{})
        payload={'model':model,'messages':messages,'temperature':params['temperature'],'top_p':params['top_p'],'max_tokens':params['num_predict']}
        if tools: payload['tools']=tools
        headers={'Authorization':f"Bearer {cfg.get('api_key','')}"} if cfg.get('api_key') else {}
        r=await client.post(base+'/chat/completions',json=payload,headers=headers); r.raise_for_status(); msg=r.json()['choices'][0]['message']
        if msg.get('tool_calls'):
            msg['tool_calls']=[{'id':t['id'],'type':'function','function':t['function']} for t in msg['tool_calls']]
        return msg

@app.post('/api/chat')
async def chat(o:Obj):
    d=o.data; sid=d.get('session_id'); text=str(d.get('message') or '').strip()
    if not sid or not text: raise HTTPException(400,'Session and message are required')
    sess=rows('SELECT * FROM sessions WHERE id=?',(sid,))
    if not sess: raise HTTPException(404,'Chat session not found')
    now=time.time(); uid=str(uuid.uuid4())
    with db() as c:
        c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(uid,sid,'user',text,None,now))
        count=c.execute('SELECT COUNT(*) FROM messages WHERE session_id=?',(sid,)).fetchone()[0]
        if count==1: c.execute('UPDATE sessions SET title=?,updated=? WHERE id=?',(text[:48],now,sid))
        else: c.execute('UPDATE sessions SET updated=? WHERE id=?',(now,sid))
    cfg=setting_map(); system=cfg.get('system_prompt',DEFAULT_SYSTEM)
    skill_context=[]
    for s in rows('SELECT * FROM skills WHERE enabled=1'):
        if s['kind']=='markdown':
            try: skill_context.append(f"\n\n## Skill: {s['name']}\n"+Path(s['path']).expanduser().read_text()[:30000])
            except Exception: pass
    messages=[{'role':'system','content':system+'\n\nEnabled skills:\n'+ '\n'.join(skill_context)}]
    history=rows('SELECT role,content FROM messages WHERE session_id=? ORDER BY created',(sid,))
    messages.extend({'role':x['role'],'content':x['content']} for x in history)
    enabled_ids={s['id'] for s in rows('SELECT id FROM skills WHERE enabled=1 AND kind="mcp"')}
    tools=[t for t in tool_catalog() if t['function']['name'].split('__',1)[0] in enabled_ids]
    trace=[]; reasoning_parts=[]
    try:
        for _ in range(8):
            answer=await model_chat(messages,tools,cfg)
            reasoning=answer.get('thinking') or answer.get('reasoning_content') or answer.get('reasoning')
            if reasoning and str(cfg.get('reasoning_enabled','false')).lower() in {'1','true','yes','on'}:
                reasoning_parts.append(str(reasoning))
            calls=answer.get('tool_calls') or []
            if not calls: break
            messages.append({'role':'assistant','content':answer.get('content') or '', 'tool_calls':calls})
            for call in calls:
                fn=call.get('function',{}); name=fn.get('name',''); raw=fn.get('arguments',{})
                if isinstance(raw,str):
                    try: raw=json.loads(raw)
                    except: raw={}
                entry={'name':name,'arguments':raw,'status':'running'}
                try:
                    validated_args=validate_mcp_tool_arguments(name,raw)
                    entry['arguments']=validated_args
                    result=await run_mcp_tool(name,validated_args); entry.update(status='success',result=result)
                    messages.append({'role':'tool','tool_call_id':call.get('id',''),'name':name,'content':json.dumps(result,ensure_ascii=False)})
                except Exception as e:
                    entry.update(status='error',result=str(e)); messages.append({'role':'tool','tool_call_id':call.get('id',''),'name':name,'content':str(e)})
                trace.append(entry)
        else: raise RuntimeError('Maximum tool-call rounds reached')
        content=answer.get('content') or '(The model returned no text.)'
        details=json.dumps({'tools':trace,'reasoning':'\n\n'.join(reasoning_parts)},ensure_ascii=False) if (trace or reasoning_parts) else None
        with db() as c: c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(str(uuid.uuid4()),sid,'assistant',content,details,time.time()))
        return {'content':content,'tools':trace,'reasoning':'\n\n'.join(reasoning_parts)}
    except httpx.HTTPError as e:
        msg=f'Model request failed: {e}'
    except Exception as e: msg=str(e)
    with db() as c: c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(str(uuid.uuid4()),sid,'assistant','Error: '+msg,json.dumps({'tools':trace,'reasoning':'\n\n'.join(reasoning_parts)},ensure_ascii=False),time.time()))
    raise HTTPException(502,msg)

@app.post('/api/shutdown')
async def shutdown():
    async def later():
        await asyncio.sleep(.25); os.kill(os.getpid(),signal.SIGTERM)
    asyncio.create_task(later()); return {'ok':True}

@app.on_event('shutdown')
async def cleanup():
    for p in list(mcp.values()):
        try: await p.stop()
        except: pass
    global model_process
    if model_process and model_process.poll() is None:
        try: model_process.terminate()
        except Exception: pass
        model_process=None
