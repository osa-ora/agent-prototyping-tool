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
from app.agent_engine import AgentBudgetExceeded, AgentRunBudget, planning_prompt, execution_context, verification_prompt, validate_plan, validate_verification, validate_tool_arguments, fallback_plan, contains_unexecuted_tool_proposal

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('AGENT_PRO_HOME', str(Path.home() / '.agent-prototyping-app')))
DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / 'agent-pro.db'
DEFAULT_SYSTEM = "I am a helpful, precise, and reliable AI agent. I follow the user's instructions, use available skills and tools when useful, explain important actions, and clearly distinguish facts from assumptions."
app = FastAPI(title='Agent Prototyping App', version='1.5.9')
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
        CREATE TABLE IF NOT EXISTS prompts(id TEXT PRIMARY KEY,name TEXT NOT NULL,description TEXT NOT NULL DEFAULT '',content TEXT NOT NULL,enabled INTEGER NOT NULL DEFAULT 1,created REAL NOT NULL,order_index INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS session_selections(session_id TEXT NOT NULL,item_type TEXT NOT NULL,item_id TEXT NOT NULL,selected INTEGER NOT NULL DEFAULT 1,PRIMARY KEY(session_id,item_type,item_id));
        CREATE TABLE IF NOT EXISTS session_selection_state(session_id TEXT PRIMARY KEY,updated REAL NOT NULL);
        ''')
        if not c.execute('SELECT 1 FROM projects LIMIT 1').fetchone():
            c.execute('INSERT INTO projects VALUES(?,?,?)',('default','Unnamed project',time.time()))
        cols={r['name'] for r in c.execute('PRAGMA table_info(skills)').fetchall()}
        if 'example_usage' not in cols: c.execute("ALTER TABLE skills ADD COLUMN example_usage TEXT NOT NULL DEFAULT ''")
        if 'transport' not in cols: c.execute("ALTER TABLE skills ADD COLUMN transport TEXT NOT NULL DEFAULT 'auto'")
        if 'endpoint' not in cols: c.execute("ALTER TABLE skills ADD COLUMN endpoint TEXT NOT NULL DEFAULT ''")
        if 'managed_locally' not in cols: c.execute("ALTER TABLE skills ADD COLUMN managed_locally INTEGER NOT NULL DEFAULT 1")
        if 'auth_headers' not in cols: c.execute("ALTER TABLE skills ADD COLUMN auth_headers TEXT NOT NULL DEFAULT '{}'")
        prompt_cols={r['name'] for r in c.execute('PRAGMA table_info(prompts)').fetchall()}
        if 'order_index' not in prompt_cols:
            c.execute('ALTER TABLE prompts ADD COLUMN order_index INTEGER NOT NULL DEFAULT 0')
            for idx, row in enumerate(c.execute('SELECT id FROM prompts ORDER BY created,id').fetchall()):
                c.execute('UPDATE prompts SET order_index=? WHERE id=?',(idx,row['id']))
        defaults={'provider':'ollama','model':'llama3.2','base_url':'http://127.0.0.1:11434','api_key':'','system_prompt':DEFAULT_SYSTEM,'temperature':'0.7','top_p':'0.9','top_k':'40','num_ctx':'8192','repeat_penalty':'1.1','max_tokens':'2048','reasoning_enabled':'false','history_limit':'all','theme':'dark','agent_architecture':'true','agent_max_llm_calls':'30','agent_max_execution_iterations':'12','agent_max_tool_calls':'20','agent_max_verification_cycles':'2','agent_max_planning_attempts':'2','agent_max_verifier_attempts':'4','agent_verifier_attempts_per_cycle':'2','agent_max_consecutive_tool_errors':'4'}
        for k,v in defaults.items(): c.execute('INSERT OR IGNORE INTO settings VALUES(?,?)',(k,v))
        # One-time cleanup of the old built-in prompt. The baseline instruction
        # now lives in settings.system_prompt, so seeding a prompt row duplicated
        # it and recreating it after deletion made user deletions non-persistent.
        if not c.execute("SELECT 1 FROM settings WHERE key='builtin_prompt_cleanup_v1'").fetchone():
            legacy_ids=[r['id'] for r in c.execute("SELECT id FROM prompts WHERE name='Default agent instructions'").fetchall()]
            for legacy_id in legacy_ids:
                c.execute("DELETE FROM session_selections WHERE item_type='prompt' AND item_id=?",(legacy_id,))
            c.execute("DELETE FROM prompts WHERE name='Default agent instructions'")
            c.execute("INSERT OR REPLACE INTO settings(key,value) VALUES('builtin_prompt_cleanup_v1','1')")

        # Repair legacy auto-selection: custom prompts must be opt-in.
        # The previous UI selected every enabled prompt for new/uninitialized chats.
        if not c.execute("SELECT 1 FROM settings WHERE key='custom_prompts_opt_in_v1'").fetchone():
            c.execute("DELETE FROM session_selections WHERE item_type='prompt' AND item_id NOT IN (SELECT id FROM prompts WHERE name='Default agent instructions')")
            baseline=c.execute("SELECT id FROM prompts WHERE name='Default agent instructions' ORDER BY created LIMIT 1").fetchone()
            if baseline:
                for session in c.execute('SELECT id FROM sessions').fetchall():
                    c.execute("INSERT OR IGNORE INTO session_selections(session_id,item_type,item_id,selected) VALUES(?, 'prompt', ?, 1)",(session['id'],baseline['id']))
            c.execute("INSERT OR REPLACE INTO settings(key,value) VALUES('custom_prompts_opt_in_v1','1')")

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
            if not bool(self.skill.get('managed_locally', 1)):
                endpoint=str(self.skill.get('endpoint') or '').strip()
                if not endpoint: raise RuntimeError('Remote MCP endpoint URL is required')
                self.transport='http'; self.endpoint=endpoint; self.proc=None; self._id=0; self.startup_error=None
                self.logs.append(f'[{time.strftime("%H:%M:%S")}] Connecting to remote MCP endpoint: {endpoint}')
                try:
                    await self.rpc('initialize', {'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'agent-prototyping-app','version':'1.2.0'}}, timeout=15)
                    await self.notify('notifications/initialized', {})
                    self.tools=(await self.rpc('tools/list',{})).get('tools',[])
                    self.logs.append(f'[{time.strftime("%H:%M:%S")}] Connected to remote MCP; discovered {len(self.tools)} tool(s)')
                    return
                except Exception as exc:
                    self.startup_error=f'{type(exc).__name__}: {exc}'
                    self.logs.append(f'[{time.strftime("%H:%M:%S")}] Remote MCP connection failed: {self.startup_error}')
                    self.tools=[]
                    raise
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
        try: headers.update(json.loads(self.skill.get('auth_headers') or '{}'))
        except Exception: pass
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
        if self.transport!='http' and (not self.proc or self.proc.returncode is not None): await self.start()
        if self.transport=='http' and not self.tools and not self.proc: await self.start()
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

def mcp_process_is_alive(skill, proc):
    if not proc:
        return False
    if not bool(skill.get('managed_locally', 1)):
        return proc.transport == 'http' and bool(proc.endpoint)
    return bool(proc.proc and proc.proc.returncode is None)

def tool_catalog(enabled_only=True):
    """Return tools from currently connected MCP servers, preserving server IDs."""
    out=[]
    for skill in rows('SELECT * FROM skills ORDER BY name'):
        if skill['kind']!='mcp' or (enabled_only and not skill['enabled']):
            continue
        proc=mcp.get(skill['id'])
        if not mcp_process_is_alive(skill, proc):
            continue
        for tool in (proc.tools if proc else []):
            out.append({'type':'function','function':{'name':f"{skill['id']}__{tool['name']}",'description':f"[{skill['name']}] {tool.get('description','')}",'parameters':tool.get('inputSchema',{'type':'object','properties':{}})}})
    return out

async def session_mcp_tools(selected_ids):
    """Refresh and build the exact MCP tool definitions for this conversation."""
    selected_ids=set(selected_ids or [])
    servers=[]
    definitions=[]
    configured=rows("SELECT * FROM skills WHERE kind='mcp' ORDER BY name")
    by_id={item['id']:item for item in configured}
    for sid in sorted(selected_ids):
        skill=by_id.get(sid)
        if not skill:
            servers.append({'id':sid,'name':sid,'status':'missing','tool_count':0})
            continue
        proc=mcp.get(sid)
        if not mcp_process_is_alive(skill,proc):
            servers.append({'id':sid,'name':skill['name'],'status':'not running','tool_count':0})
            continue
        try:
            # Refresh discovery for every chat request. This prevents a stale or empty
            # in-memory catalog from silently omitting tools from the model request.
            discovered=await proc.rpc('tools/list',{},timeout=10)
            proc.tools=discovered.get('tools',[]) or []
            status='running'
        except Exception as exc:
            servers.append({'id':sid,'name':skill['name'],'status':f'discovery failed: {type(exc).__name__}: {exc}','tool_count':0})
            continue
        servers.append({'id':sid,'name':skill['name'],'status':status,'tool_count':len(proc.tools), 'tools':[t.get('name') for t in proc.tools]})
        for tool in proc.tools:
            if not tool.get('name'):
                continue
            definitions.append({'type':'function','function':{'name':f"{sid}__{tool['name']}",'description':f"[{skill['name']}] {tool.get('description','')}",'parameters':tool.get('inputSchema',{'type':'object','properties':{}})}})
    return definitions,servers

@app.get('/',response_class=HTMLResponse)
async def home(): return (ROOT/'app'/'templates'/'index.html').read_text()
@app.get('/api/health')
async def health(): return {'ok':True,'version':'1.5.9'}
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
    skills=rows('SELECT id,name,kind,path,enabled,command,args,env,example_usage,transport,endpoint,managed_locally,auth_headers FROM skills ORDER BY created')
    for s in skills:
        p=mcp.get(s['id']); s['running']=bool(p and mcp_process_is_alive(s,p) and p.tools); s['starting']=bool(p and p.starting); s['error']=p.startup_error if p else None; s['tools']=p.tools if s['running'] else []
    cfg=setting_map(); provider=cfg.get('provider','ollama'); running=False; models=[]
    if provider=='ollama':
        try:
            running,models=await ollama_http_status()
        except Exception: pass
    prompts=rows('SELECT * FROM prompts ORDER BY order_index,created,id')
    return {'projects':projects,'sessions':sessions,'skills':skills,'prompts':prompts,'settings':cfg,'model_running':running,'model_owned':bool(model_process and model_process.poll() is None),'models':models}

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
    if kind=='markdown':
        pth=Path(path).expanduser() if path else None
        if not pth or not pth.exists() or not (pth.is_file() or (pth.is_dir() and (pth/'SKILL.md').is_file())): raise HTTPException(400,'Select a skill directory containing SKILL.md, or a Markdown file')
    managed=bool(d.get('managed_locally',True))
    if kind=='mcp' and managed and not str(d.get('command') or '').strip(): raise HTTPException(400,'Local MCP start command is required')
    if kind=='mcp' and not managed and not str(d.get('endpoint') or '').strip(): raise HTTPException(400,'Remote MCP endpoint URL is required')
    transport=str(d.get('transport') or 'auto').lower()
    if transport not in {'auto','stdio','http'}: raise HTTPException(400,'Transport must be Auto, STDIO, or Streamable HTTP')
    endpoint=str(d.get('endpoint') or '').strip()
    if kind=='mcp' and (not managed or transport=='http') and not endpoint: raise HTTPException(400,'Enter the Streamable HTTP MCP endpoint URL')
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
    with db() as c: c.execute('INSERT INTO skills(id,name,kind,path,enabled,command,args,env,example_usage,created,transport,endpoint,managed_locally,auth_headers) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(sid,name,kind,path,1 if d.get('enabled',True) else 0,d.get('command',''),json.dumps(args),json.dumps(env),str(d.get('example_usage') or '').strip(),time.time(),transport,endpoint,int(managed),json.dumps(d.get('auth_headers') or {})))
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
    for key in ('path','command','transport','endpoint','managed_locally','auth_headers'):
        if key in d and str(d[key] or '') != str(s.get(key) or ''): launch_changed = True
    normalized={}
    for key in ('args','env'):
        if key in d:
            normalized[key]=parse_mcp_args(d[key]) if key=='args' else parse_mcp_env(d[key])
            existing=json.loads(s.get(key) or ('[]' if key=='args' else '{}'))
            if json.dumps(normalized[key],sort_keys=True) != json.dumps(existing,sort_keys=True): launch_changed=True
    was_running=bool(sid in mcp and mcp[sid].tools and ((not s.get('managed_locally',1) and mcp[sid].transport=='http') or (mcp[sid].proc and mcp[sid].proc.returncode is None)))
    if launch_changed and sid in mcp:
        await mcp[sid].stop(); mcp.pop(sid,None)
    with db() as c:
        for k in ('name','path','command','example_usage','transport','endpoint'):
            if k in d: c.execute(f'UPDATE skills SET {k}=? WHERE id=?',(str(d[k]),sid))
        if 'managed_locally' in d: c.execute('UPDATE skills SET managed_locally=? WHERE id=?',(int(bool(d['managed_locally'])),sid))
        for k in ('args','env'):
            if k in normalized: c.execute(f'UPDATE skills SET {k}=? WHERE id=?',(json.dumps(normalized[k]),sid))
        if 'auth_headers' in d: c.execute('UPDATE skills SET auth_headers=? WHERE id=?',(json.dumps(d['auth_headers']),sid))
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
        return {'ok':True,'tools':p.tools,'running':bool(p.tools and ((not p.skill.get('managed_locally',1) and p.transport=='http') or (p.proc and p.proc.returncode is None))),'transport':p.transport,'endpoint':p.endpoint}
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
    return {'logs':list(p.logs) if p else ['No process has been started in this application session yet.'], 'running':bool(p and p.tools and ((not s.get('managed_locally',1) and p.transport=='http') or (p.proc and p.proc.returncode is None))), 'transport':p.transport if p else None, 'endpoint':p.endpoint if p else None}

@app.post('/api/skills/{sid}/refresh')
async def refresh_skill(sid:str):
    if sid not in mcp: raise HTTPException(400,'Start the MCP server first')
    p=mcp[sid]
    try: p.tools=(await p.rpc('tools/list',{})).get('tools',[]); return {'tools':p.tools}
    except Exception as e: raise HTTPException(500,str(e))

@app.post('/api/prompts')
async def add_prompt(o:Obj):
    d=o.data; name=str(d.get('name') or '').strip(); content=str(d.get('content') or '').strip()
    if not name or not content: raise HTTPException(400,'Prompt name and content are required')
    pid=str(uuid.uuid4())
    with db() as c:
        next_order=c.execute('SELECT COALESCE(MAX(order_index),-1)+1 FROM prompts').fetchone()[0]
        c.execute('INSERT INTO prompts(id,name,description,content,enabled,created,order_index) VALUES(?,?,?,?,?,?,?)',(pid,name,str(d.get('description') or ''),content,int(bool(d.get('enabled',True))),time.time(),next_order))
    return {'id':pid}
@app.patch('/api/prompts/{pid}')
async def update_prompt(pid:str,o:Obj):
    if not rows('SELECT id FROM prompts WHERE id=?',(pid,)): raise HTTPException(404,'Prompt not found')
    d=o.data
    with db() as c:
        for k in ('name','description','content'):
            if k in d: c.execute(f'UPDATE prompts SET {k}=? WHERE id=?',(str(d[k]),pid))
        if 'enabled' in d: c.execute('UPDATE prompts SET enabled=? WHERE id=?',(int(bool(d['enabled'])),pid))
    return {'ok':True}
@app.put('/api/prompts/order')
async def reorder_prompts(o:Obj):
    ids=o.data.get('ids') or []
    if not isinstance(ids,list) or any(not isinstance(x,str) for x in ids): raise HTTPException(400,'Prompt order must be a list of IDs')
    with db() as c:
        existing=[r['id'] for r in c.execute('SELECT id FROM prompts ORDER BY order_index,created,id').fetchall()]
        if len(ids)!=len(existing) or set(ids)!=set(existing): raise HTTPException(400,'Prompt order must include every prompt exactly once')
        for index,pid in enumerate(ids): c.execute('UPDATE prompts SET order_index=? WHERE id=?',(index,pid))
    return {'ok':True}
@app.delete('/api/prompts/{pid}')
async def delete_prompt(pid:str):
    with db() as c:
        c.execute('DELETE FROM prompts WHERE id=?',(pid,)); c.execute("DELETE FROM session_selections WHERE item_type='prompt' AND item_id=?",(pid,))
    return {'ok':True}
@app.get('/api/sessions/{sid}/selections')
async def get_selections(sid:str):
    if not rows('SELECT id FROM sessions WHERE id=?',(sid,)): raise HTTPException(404,'Session not found')
    return {'prompts': [x['item_id'] for x in rows("SELECT item_id FROM session_selections WHERE session_id=? AND item_type='prompt' AND selected=1",(sid,))], 'skills': [x['item_id'] for x in rows("SELECT item_id FROM session_selections WHERE session_id=? AND item_type='skill' AND selected=1",(sid,))], 'mcp': [x['item_id'] for x in rows("SELECT item_id FROM session_selections WHERE session_id=? AND item_type='mcp' AND selected=1",(sid,))], 'initialized': bool(rows('SELECT 1 FROM session_selection_state WHERE session_id=?',(sid,)))}
@app.put('/api/sessions/{sid}/selections')
async def put_selections(sid:str,o:Obj):
    if not rows('SELECT id FROM sessions WHERE id=?',(sid,)): raise HTTPException(404,'Session not found')
    d=o.data
    with db() as c:
        c.execute('INSERT INTO session_selection_state(session_id,updated) VALUES(?,?) ON CONFLICT(session_id) DO UPDATE SET updated=excluded.updated',(sid,time.time()))
        for kind,key in [('prompt','prompts'),('skill','skills'),('mcp','mcp')]:
            chosen=set(str(x) for x in d.get(key,[]) if x)
            c.execute('DELETE FROM session_selections WHERE session_id=? AND item_type=?',(sid,kind))
            for item_id in chosen: c.execute('INSERT OR REPLACE INTO session_selections VALUES(?,?,?,1)',(sid,kind,item_id))
    return {'ok':True}

@app.put('/api/settings')
async def save_settings(o:Obj):
    allowed={'provider','model','base_url','api_key','system_prompt','temperature','top_p','top_k','num_ctx','repeat_penalty','max_tokens','reasoning_enabled','history_limit','theme','agent_architecture'}
    agent_integer_ranges={
        'agent_max_llm_calls':(1,200),
        'agent_max_execution_iterations':(1,100),
        'agent_max_tool_calls':(1,200),
        'agent_max_verification_cycles':(1,20),
        'agent_max_planning_attempts':(1,10),
        'agent_max_verifier_attempts':(1,50),
        'agent_verifier_attempts_per_cycle':(1,10),
        'agent_max_consecutive_tool_errors':(1,30),
    }
    updates={}
    for k,v in o.data.items():
        if k in agent_integer_ranges:
            try: number=int(v)
            except (TypeError,ValueError): raise HTTPException(400,f'{k} must be an integer')
            low,high=agent_integer_ranges[k]
            if number<low or number>high: raise HTTPException(400,f'{k} must be between {low} and {high}')
            updates[k]=str(number)
        elif k=='agent_architecture':
            if str(v).lower() not in {'1','0','true','false','yes','no','on','off'}:
                raise HTTPException(400,'agent_architecture must be a boolean')
            updates[k]='true' if str(v).lower() in {'1','true','yes','on'} else 'false'
        elif k in allowed:
            updates[k]=str(v)
    with db() as c:
        for k,v in updates.items(): c.execute('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(k,v))
    return {'ok':True,'updated':sorted(updates)}
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
    if running: return {'ok':True,'already_running':True,'usage_reset':True}
    if model_process and model_process.poll() is None: return {'ok':True,'already_running':True,'starting':True,'usage_reset':True}
    executable=find_ollama_executable()
    if not executable: raise HTTPException(500,'Ollama executable was not found. Install Ollama or add it to PATH.')
    try:
        model_process=subprocess.Popen([executable,'serve'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        return {'ok':True,'already_running':False,'starting':True,'pid':model_process.pid,'usage_reset':True}
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
        return {'ok':True,'stopped':False,'externally_managed':True,'usage_reset':True,'message':f'Ollama is responding but macOS did not allow process inspection ({e}). Close it from the Ollama menu-bar app to stop it.'}
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
        return {'ok':True,'stopped':True,'pids':stopped,'usage_reset':True}
    if not stopped:
        return {'ok':True,'stopped':False,'externally_managed':True,'usage_reset':True,'message':'Ollama is managed by another process and is still responding.'}
    return {'ok':True,'stopped':False,'externally_managed':True,'pids':stopped,'usage_reset':True,'message':'A stop signal was sent, but Ollama is still responding.'}



async def run_mcp_tool(full_name,args):
    sid,name=full_name.split('__',1); p=mcp.get(sid)
    connected=bool(p and p.tools and ((not p.skill.get('managed_locally',1) and p.transport=='http') or (p.proc and p.proc.returncode is None)))
    if not connected: raise RuntimeError('MCP server is not connected or running')
    return await p.call(name,args)

def normalize_mcp_result(result):
    """Convert the MCP CallToolResult envelope into concise model-facing content."""
    if not isinstance(result, dict):
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if result.get('isError'):
        parts=[]
        for item in result.get('content') or []:
            if isinstance(item, dict) and item.get('text'):
                parts.append(str(item['text']))
        detail='\n'.join(parts).strip() or 'MCP tool reported an error.'
        raise RuntimeError(detail)
    content=result.get('content')
    if isinstance(content, list):
        parts=[]
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item)); continue
            if item.get('type') == 'text' and item.get('text') is not None:
                parts.append(str(item['text']))
            elif item.get('type') == 'resource' and item.get('resource') is not None:
                parts.append(json.dumps(item['resource'], ensure_ascii=False))
            elif item.get('type') not in {'image','audio','resource'}:
                parts.append(json.dumps(item, ensure_ascii=False))
        if parts:
            return '\n'.join(parts).strip()
    structured=result.get('structuredContent')
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    # Some MCP servers return a plain JSON object rather than a CallToolResult.
    return json.dumps(result, ensure_ascii=False)

async def model_chat(messages, tools, cfg):
    started=time.perf_counter()
    provider=cfg.get('provider','ollama'); base=cfg.get('base_url','http://127.0.0.1:11434').rstrip('/'); model=cfg.get('model','llama3.2')
    params={'temperature':float(cfg.get('temperature',.7)),'top_p':float(cfg.get('top_p',.9)),'top_k':int(cfg.get('top_k',40)),'repeat_penalty':float(cfg.get('repeat_penalty',1.1)),'num_ctx':int(cfg.get('num_ctx',8192)),'num_predict':int(cfg.get('max_tokens',2048))}
    async with httpx.AsyncClient(timeout=180) as client:
        if provider=='ollama':
            payload={'model':model,'messages':messages,'stream':False,'options':params}
            if str(cfg.get('reasoning_enabled','false')).lower() in {'1','true','yes','on'}: payload['think']=True
            if cfg.get('_response_format') == 'json': payload['format']='json'
            if tools: payload['tools']=tools
            r=await client.post(base+'/api/chat',json=payload); r.raise_for_status(); data=r.json()
            input_tokens=data.get('prompt_eval_count'); output_tokens=data.get('eval_count')
            usage={'input_tokens':input_tokens,'output_tokens':output_tokens,'total_tokens':(input_tokens+output_tokens) if isinstance(input_tokens,int) and isinstance(output_tokens,int) else None}
            # Ollama reports these durations in nanoseconds. With non-streaming chat,
            # TTFT is an estimate (model load + prompt evaluation), not a measured token timestamp.
            load_ns=data.get('load_duration'); prompt_ns=data.get('prompt_eval_duration'); eval_ns=data.get('eval_duration')
            ttft_ns=(load_ns+prompt_ns) if isinstance(load_ns,(int,float)) and isinstance(prompt_ns,(int,float)) else None
            itl_ms=(eval_ns/1_000_000/(output_tokens-1)) if isinstance(eval_ns,(int,float)) and isinstance(output_tokens,int) and output_tokens>1 else None
            tokens_per_second=(output_tokens/(eval_ns/1_000_000_000)) if isinstance(eval_ns,(int,float)) and eval_ns>0 and isinstance(output_tokens,int) else None
            timing={'load_duration_ms':load_ns/1_000_000 if isinstance(load_ns,(int,float)) else None,'prompt_eval_duration_ms':prompt_ns/1_000_000 if isinstance(prompt_ns,(int,float)) else None,'generation_duration_ms':eval_ns/1_000_000 if isinstance(eval_ns,(int,float)) else None,'ttft_ms':ttft_ns/1_000_000 if ttft_ns is not None else None,'ttft_estimated':ttft_ns is not None,'itl_ms':itl_ms,'tokens_per_second':tokens_per_second,'total_duration_ms':data.get('total_duration')/1_000_000 if isinstance(data.get('total_duration'),(int,float)) else None}
            return {'message':data.get('message',{}),'usage':usage,'timing':timing,'duration_ms':round((time.perf_counter()-started)*1000),'model':model,'provider':provider,'request':{'endpoint':base+'/api/chat','payload':payload}}
        payload={'model':model,'messages':messages,'temperature':params['temperature'],'top_p':params['top_p'],'max_tokens':params['num_predict']}
        if cfg.get('_response_format') == 'json': payload['response_format']={'type':'json_object'}
        if tools: payload['tools']=tools
        headers={'Authorization':f"Bearer {cfg.get('api_key','')}"} if cfg.get('api_key') else {}
        r=await client.post(base+'/chat/completions',json=payload,headers=headers); r.raise_for_status(); data=r.json(); msg=data['choices'][0]['message']; raw_usage=data.get('usage') or {}; usage={'input_tokens':raw_usage.get('prompt_tokens'),'output_tokens':raw_usage.get('completion_tokens'),'total_tokens':raw_usage.get('total_tokens')}; usage['total_tokens']=usage['input_tokens']+usage['output_tokens'] if usage.get('total_tokens') is None and isinstance(usage.get('input_tokens'),int) and isinstance(usage.get('output_tokens'),int) else usage.get('total_tokens')
        if msg.get('tool_calls'):
            msg['tool_calls']=[{'id':t['id'],'type':'function','function':t['function']} for t in msg['tool_calls']]
        return {'message':msg,'usage':usage,'timing':{'load_duration_ms':None,'prompt_eval_duration_ms':None,'generation_duration_ms':None,'ttft_ms':None,'ttft_estimated':False,'itl_ms':None,'tokens_per_second':None,'total_duration_ms':None},'duration_ms':round((time.perf_counter()-started)*1000),'model':model,'provider':provider,'request':{'endpoint':base+'/chat/completions','payload':payload}}

@app.post('/api/chat')
async def chat(o:Obj):
    d=o.data; sid=d.get('session_id'); text=str(d.get('message') or '').strip()
    if not sid or not text: raise HTTPException(400,'Session and message are required')
    sess=rows('SELECT * FROM sessions WHERE id=?',(sid,))
    if not sess: raise HTTPException(404,'Chat session not found')
    # The chat request carries the exact checkbox state currently visible in the UI.
    # Persist it synchronously before resolving MCP servers, eliminating races between
    # the checkbox's asynchronous save and an immediately submitted chat message.
    submitted_selections=d.get('selections')
    if isinstance(submitted_selections,dict):
        with db() as c:
            c.execute('INSERT INTO session_selection_state(session_id,updated) VALUES(?,?) ON CONFLICT(session_id) DO UPDATE SET updated=excluded.updated',(sid,time.time()))
            for kind,key in [('prompt','prompts'),('skill','skills'),('mcp','mcp')]:
                chosen={str(x) for x in (submitted_selections.get(key) or []) if x}
                c.execute('DELETE FROM session_selections WHERE session_id=? AND item_type=?',(sid,kind))
                for item_id in chosen:
                    c.execute('INSERT OR REPLACE INTO session_selections(session_id,item_type,item_id,selected) VALUES(?,?,?,1)',(sid,kind,item_id))
    now=time.time(); uid=str(uuid.uuid4())
    with db() as c:
        c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(uid,sid,'user',text,None,now))
        count=c.execute('SELECT COUNT(*) FROM messages WHERE session_id=?',(sid,)).fetchone()[0]
        if count==1: c.execute('UPDATE sessions SET title=?,updated=? WHERE id=?',(text[:48],now,sid))
        else: c.execute('UPDATE sessions SET updated=? WHERE id=?',(now,sid))
    cfg=setting_map(); system=cfg.get('system_prompt',DEFAULT_SYSTEM)
    selection={r['item_type']:{x['item_id'] for x in rows('SELECT item_id FROM session_selections WHERE session_id=? AND item_type=? AND selected=1',(sid,r['item_type']))} for r in rows('SELECT DISTINCT item_type FROM session_selections WHERE session_id=?',(sid,))}
    selected_prompts=selection.get('prompt',set()); selected_skills=selection.get('skill',set()); selected_mcp=selection.get('mcp',set())
    # Legacy conversations without an explicit selection record inherit enabled
    # skills and MCP servers, but prompts are opt-in except for the built-in baseline.
    has_selection=bool(rows('SELECT 1 FROM session_selection_state WHERE session_id=?',(sid,)))
    if not has_selection:
        selected_prompts={x['id'] for x in rows("SELECT id FROM prompts WHERE enabled=1 AND name='Default agent instructions'")}
        selected_skills={x['id'] for x in rows("SELECT id FROM skills WHERE enabled=1 AND kind='markdown'")}
        selected_mcp={x['id'] for x in rows("SELECT id FROM skills WHERE enabled=1 AND kind='mcp'")}
    # Prompt names and section labels are UI metadata, not instructions for the
    # model. Send only the actual selected prompt bodies.
    prompt_context=[str(p['content']).strip() for p in rows('SELECT * FROM prompts WHERE enabled=1 ORDER BY order_index,created,id') if p['id'] in selected_prompts and str(p.get('content') or '').strip()]
    skill_context=[]
    for sk in rows("SELECT * FROM skills WHERE enabled=1 AND kind='markdown'"):
        if sk['id'] not in selected_skills: continue
        try:
            root=Path(sk['path']).expanduser()
            skill_file=root/'SKILL.md' if root.is_dir() else root
            content=skill_file.read_text()[:30000]
            if root.is_dir():
                for sub in ('references','prompts'):
                    folder=root/sub
                    if folder.is_dir():
                        for resource in sorted(folder.glob('*.md'))[:12]: content += f"\n\n### {resource.relative_to(root)}\n"+resource.read_text(errors='replace')[:10000]
            skill_context.append(content.strip())
        except Exception: pass
    # Build a clean system message: configured system instructions (if any),
    # followed by the literal selected prompt and skill contents. Never inject
    # headings such as 'Selected prompts' or prompt names into the model context.
    system_parts=[str(system).strip()] if str(system or '').strip() else []
    system_parts.extend(prompt_context)
    system_parts.extend(skill_context)
    messages=[]
    if system_parts:
        messages.append({'role':'system','content':'\n\n'.join(system_parts)})
    # History retention applies only to messages from earlier user requests.
    # The current request is appended separately and is never counted against
    # the configured cross-request history window.
    history=rows('SELECT role,content FROM messages WHERE session_id=? AND id<>? ORDER BY created,id',(sid,uid))
    history_limit=str(cfg.get('history_limit','all')).strip().lower()
    if history_limit != 'all':
        try:
            limit=int(history_limit)
            if limit in {0,1,2,4,6,8,10}:
                history=history[-limit:] if limit else []
        except (TypeError,ValueError):
            pass
    messages.extend({'role':x['role'],'content':x['content']} for x in history)
    messages.append({'role':'user','content':text})
    # MCP availability is controlled by explicit per-conversation selection and
    # server connection state, not the legacy global skill-enabled flag.
    enabled_ids=selected_mcp if has_selection else set()
    tools, mcp_context = await session_mcp_tools(enabled_ids)
    tool_schemas = {item.get('function', {}).get('name'): item.get('function', {}).get('parameters', {'type':'object','properties':{}}) for item in tools}
    # Do not silently send a tool-less request when the user selected MCP servers.
    # A visible error is safer than allowing the model to hallucinate tool results.
    if enabled_ids and not tools:
        diagnostics='; '.join(f"{x['name']}: {x['status']} ({x['tool_count']} tools)" for x in mcp_context)
        raise HTTPException(409, 'Selected MCP servers supplied no tools to the model. ' + (diagnostics or 'No selected server could be queried.'))
    trace=[]; reasoning_parts=[]; events=[]; usage_totals={'input_tokens':0,'output_tokens':0,'total_tokens':0}; usage_complete=True; llm_index=0
    task_plan=None; completion_verification=None; completion_status='unverified'
    run_budget=AgentRunBudget(
        max_llm_calls=int(cfg.get('agent_max_llm_calls',30)),
        max_execution_iterations=int(cfg.get('agent_max_execution_iterations',12)),
        max_tool_calls=int(cfg.get('agent_max_tool_calls',20)),
        max_verification_cycles=int(cfg.get('agent_max_verification_cycles',2)),
        max_verifier_attempts=int(cfg.get('agent_max_verifier_attempts',4)),
        max_consecutive_tool_errors=int(cfg.get('agent_max_consecutive_tool_errors',4)),
    )
    try:
        architecture_enabled=str(cfg.get('agent_architecture','true')).lower() in {'1','true','yes','on'}
        if architecture_enabled:
            # Phase 1: ask the configured model to decompose the request into
            # generic, assessable objectives. The planner sees capability
            # descriptions but receives no executable tools.
            planner_cfg=dict(cfg); planner_cfg['_response_format']='json'
            plan_messages=list(messages)
            plan_message={'role':'system','content':planning_prompt(tools)}
            plan_insert=1 if plan_messages and plan_messages[0].get('role')=='system' else 0
            plan_messages.insert(plan_insert,plan_message)
            plan_validation_error=None
            for plan_attempt in range(max(1, int(cfg.get('agent_max_planning_attempts',2)))):
                if plan_validation_error:
                    plan_messages.append({'role':'system','content':f'The previous planning response did not satisfy the required JSON structure: {plan_validation_error}. Correct it and return only a valid JSON object matching the requested plan schema.'})
                run_budget.consume('llm_calls')
                plan_invocation=await model_chat(plan_messages,[],planner_cfg)
                plan_answer=plan_invocation.get('message') or {}
                llm_index+=1
                plan_usage=plan_invocation.get('usage') or {}
                for key in ('input_tokens','output_tokens'):
                    value=plan_usage.get(key)
                    if isinstance(value,int): usage_totals[key]+=value
                    else: usage_complete=False
                plan_event={'type':'llm','phase':'planning','attempt':plan_attempt+1,'index':llm_index,'provider':plan_invocation.get('provider'),'model':plan_invocation.get('model'),'duration_ms':plan_invocation.get('duration_ms'),'usage':plan_usage,'timing':plan_invocation.get('timing') or {},'tool_selection':False,'completion_check':False,'mcp_servers':mcp_context,'exposed_tool_names':[],'request':plan_invocation.get('request') or {},'model_response':plan_answer,'response':plan_answer.get('content') or ''}
                events.append(plan_event)
                try:
                    if plan_answer.get('tool_calls'):
                        raise ValueError('The task planner attempted a tool call; planning must not execute tools.')
                    task_plan=validate_plan(plan_answer.get('content'))
                    break
                except ValueError as exc:
                    plan_validation_error=str(exc)
                    plan_event['validation_error']=plan_validation_error
                    if plan_attempt + 1 >= max(1, int(cfg.get('agent_max_planning_attempts',2))):
                        # Planning is advisory. A malformed planner response must
                        # not prevent the executor from handling a clear request.
                        task_plan=fallback_plan(text, plan_validation_error)
                        events.append({
                            'type':'agent_fallback', 'phase':'planning',
                            'strategy':'minimal_request_derived_plan',
                            'reason':plan_validation_error,
                            'plan':task_plan,
                        })
            plan_context=execution_context(task_plan)
            insert_at=1 if messages and messages[0].get('role')=='system' else 0
            messages.insert(insert_at,{'role':'system','content':plan_context})

            # Phase 2: model-directed execution. Every native tool call is
            # executed and acknowledged; only the model chooses subsequent work.
            answer=None; candidate_answer=None
            for round_index in range(run_budget.limits['execution_iterations']):
                run_budget.consume('execution_iterations')
                run_budget.consume('llm_calls')
                invocation=await model_chat(messages,tools,cfg); answer=invocation['message']; llm_index+=1
                usage=invocation.get('usage') or {}
                for key in ('input_tokens','output_tokens'):
                    value=usage.get(key)
                    if isinstance(value,int): usage_totals[key]+=value
                    else: usage_complete=False
                events.append({'type':'llm','phase':'execution','round':round_index+1,'index':llm_index,'provider':invocation.get('provider'),'model':invocation.get('model'),'duration_ms':invocation.get('duration_ms'),'usage':usage,'timing':invocation.get('timing') or {},'tool_selection':bool(answer.get('tool_calls')),'completion_check':False,'mcp_servers':mcp_context,'exposed_tool_names':[x['function']['name'] for x in tools],'request':invocation.get('request') or {},'model_response':answer,'response':answer.get('content') or ''})
                reasoning=answer.get('thinking') or answer.get('reasoning_content') or answer.get('reasoning')
                if reasoning and str(cfg.get('reasoning_enabled','false')).lower() in {'1','true','yes','on'}:
                    reasoning_parts.append(str(reasoning))
                calls=answer.get('tool_calls') or []
                if calls:
                    messages.append({'role':'assistant','content':answer.get('content') or '', 'tool_calls':calls})
                    candidate_answer=None
                    for call in calls:
                        run_budget.consume('tool_calls')
                        fn=call.get('function',{}) if isinstance(call,dict) else {}; fn=fn if isinstance(fn,dict) else {}; name=fn.get('name',''); raw=fn.get('arguments',{})
                        if isinstance(raw,str):
                            try: raw=json.loads(raw); raw_error=None
                            except json.JSONDecodeError as exc: raw_error=f'Tool arguments are not valid JSON: {exc}'
                        else: raw_error=None
                        call_id=call.get('id','') if isinstance(call,dict) else ''
                        entry={'name':name or 'Invalid native tool call','call_id':call_id,'arguments':raw,'status':'running'}; tool_started=time.perf_counter()
                        try:
                            if not call_id: raise ValueError('The native tool call is missing its required call ID.')
                            if not name: raise ValueError('The native tool call is missing its function name.')
                            if raw_error: raise ValueError(raw_error)
                            if not isinstance(raw,dict): raise ValueError('Tool arguments must be a JSON object.')
                            if name not in tool_schemas: raise ValueError(f'The model requested a tool that was not in the advertised tool catalog: {name}')
                            validate_tool_arguments(raw, tool_schemas[name])
                            result=await run_mcp_tool(name,raw)
                            model_result=normalize_mcp_result(result)
                            entry.update(status='success',result=result)
                            messages.append({'role':'tool','tool_call_id':call_id,'name':name,'content':model_result if isinstance(model_result,str) else json.dumps(model_result,ensure_ascii=False)})
                        except Exception as e:
                            entry.update(status='error',result=str(e));
                            if call_id: messages.append({'role':'tool','tool_call_id':call_id,'name':name or 'invalid_tool_call','content':str(e)})
                            else: messages.append({'role':'system','content':'A malformed native tool call was rejected: ' + str(e)})
                        entry['duration_ms']=round((time.perf_counter()-tool_started)*1000); trace.append(entry); events.append({'type':'tool','name':entry['name'],'call_id':call_id,'arguments':entry.get('arguments',raw),'status':entry.get('status'),'result':entry.get('result'),'duration_ms':entry['duration_ms']})
                        run_budget.record_tool_result(entry.get('status') == 'success')
                    continue

                candidate_answer=str(answer.get('content') or '').strip()
                if contains_unexecuted_tool_proposal(candidate_answer, tools):
                    # The model described a tool invocation in prose instead of
                    # issuing a native structured call. Never parse or execute it.
                    # Keep the transcript and ask the model to retry through the
                    # actual tool-calling protocol before considering completion.
                    messages.append({'role':'assistant','content':candidate_answer})
                    messages.append({'role':'system','content':
                        'Your previous response described a call to an available tool as ordinary text. '
                        'That did not execute the tool and its proposed arguments are not evidence. '
                        'If the request requires that operation, issue it now using a native structured '
                        'tool_calls response with the exact advertised function name and schema-valid '
                        'arguments. Do not print JSON or a proposed call in assistant text. If no tool '
                        'is required, answer using only the evidence already available.'})
                    events.append({'type':'agent_recovery','phase':'execution','reason':'unexecuted_tool_proposal_in_assistant_text','action':'request_native_structured_tool_call'})
                    candidate_answer=None
                    continue
                run_budget.consume('verification_cycles')
                messages.append({'role':'assistant','content':candidate_answer})
                # Phase 3: independent completion verification. The verifier
                # receives the entire transcript but no executable tools.
                verifier_cfg=dict(cfg); verifier_cfg['_response_format']='json'
                verifier_messages=list(messages)
                successful_tool_calls = [
                    {'call_id': x.get('call_id', ''), 'name': x.get('name', '')}
                    for x in trace
                    if x.get('status') == 'success' and x.get('call_id')
                ]
                successful_tool_names = {}
                for item in successful_tool_calls:
                    successful_tool_names.setdefault(item['name'], []).append(item['call_id'])
                verifier_instruction={'role':'system','content':verification_prompt(task_plan,candidate_answer,successful_tool_calls)}
                verifier_insert=1 if verifier_messages and verifier_messages[0].get('role')=='system' else 0
                verifier_messages.insert(verifier_insert,verifier_instruction)
                verification_error=None
                verification_valid=False
                if run_budget.used['verifier_attempts'] >= run_budget.limits['verifier_attempts']:
                    raise AgentBudgetExceeded('Agent run stopped: verifier attempts budget exhausted (' + str(run_budget.used['verifier_attempts']) + '/' + str(run_budget.limits['verifier_attempts']) + ' used).')
                verifier_attempts_per_cycle=max(1, int(cfg.get('agent_verifier_attempts_per_cycle',2)))
                # Initial assessment plus a bounded number of corrections. Each
                # correction sees the exact invalid response and validation error.
                for verification_attempt in range(min(verifier_attempts_per_cycle, run_budget.limits['verifier_attempts'] - run_budget.used['verifier_attempts'])):
                    if verification_error:
                        verifier_messages.append({'role':'assistant','content':str(verification_answer.get('content') or '')})
                        verifier_messages.append({'role':'system','content':
                            'Your previous assessment failed structural or evidence validation. '
                            'The exact validation error is: ' + verification_error + '. '
                            'Return a corrected JSON object matching the required schema. '
                            'Reassess the evidence; do not invent evidence or weaken requirements.'})
                        # The prior error belongs to the previous response only.
                        # Clear it before evaluating the new model response; otherwise
                        # a valid correction is incorrectly skipped and every retry
                        # is treated as a failure.
                        verification_error=None
                    run_budget.consume('verifier_attempts')
                    run_budget.consume('llm_calls')
                    verification_invocation=await model_chat(verifier_messages,[],verifier_cfg)
                    verification_answer=verification_invocation.get('message') or {}
                    if verification_answer.get('tool_calls'):
                        verification_error='The verifier attempted a tool call; verification must not execute tools.'
                    llm_index+=1
                    verification_usage=verification_invocation.get('usage') or {}
                    for key in ('input_tokens','output_tokens'):
                        value=verification_usage.get(key)
                        if isinstance(value,int): usage_totals[key]+=value
                        else: usage_complete=False
                    verification_event={'type':'llm','phase':'verification','round':round_index+1,'attempt':verification_attempt+1,'index':llm_index,'provider':verification_invocation.get('provider'),'model':verification_invocation.get('model'),'duration_ms':verification_invocation.get('duration_ms'),'usage':verification_usage,'timing':verification_invocation.get('timing') or {},'tool_selection':False,'completion_check':True,'mcp_servers':mcp_context,'exposed_tool_names':[],'request':verification_invocation.get('request') or {},'model_response':verification_answer,'response':verification_answer.get('content') or ''}
                    events.append(verification_event)
                    if not verification_error:
                        try:
                            completion_verification=validate_verification(verification_answer.get('content'),task_plan,{x.get('call_id') for x in trace if x.get('status')=='success' and x.get('call_id')},len(messages),successful_tool_names)
                            verification_event['verification']=completion_verification
                            verification_valid=True
                            break
                        except ValueError as exc:
                            verification_error=str(exc)
                    verification_event['validation_error']=verification_error
                if not verification_valid:
                    # A verifier protocol failure is not evidence that execution is
                    # incomplete. Retrying the executor cannot repair a verifier's
                    # malformed JSON/status and caused repeated, wasteful cycles.
                    # Preserve the candidate but explicitly leave it unverified.
                    completion_status='unverified'
                    completion_verification={
                        'status':'unverified',
                        'reason':'The independent verifier did not return a valid assessment after its correction attempt.',
                        'attempts_this_cycle':verification_attempt + 1,
                        'total_attempts':run_budget.used['verifier_attempts'],
                        'last_validation_error':str(verification_error),
                    }
                    verification_event['verification_fallback']='candidate_preserved_unverified'
                    events.append({'type':'agent_recovery','phase':'verification','reason':str(verification_error),'action':'stop_without_reexecuting_tools'})
                    answer={'role':'assistant','content':candidate_answer + '\n\n(Independent verification could not be completed; the retrieved information is presented as unverified.)'}
                    break
                if completion_verification['status']=='complete':
                    completion_status='complete'
                    answer={'role':'assistant','content':candidate_answer}
                    break
                if completion_verification['status']=='needs_clarification':
                    completion_status='needs_clarification'
                    final_text=candidate_answer or completion_verification.get('clarification_question') or 'I need more information to complete this request.'
                    if completion_verification.get('clarification_question') and completion_verification.get('clarification_question') not in final_text:
                        final_text=final_text+'\n\n'+completion_verification['clarification_question']
                    answer={'role':'assistant','content':final_text}
                    break
                completion_status='incomplete'
                # Feed the verifier's concrete gaps back to the executor. This
                # is feedback, not an application-selected tool or workflow.
                feedback=(
                    'Independent completion verification found that the task is not complete. '
                    'Reassess the original request and the full transcript, then decide what to do next. '
                    'Use available native tools when they can close a gap; otherwise explain the limitation. '
                    'Do not repeat the same unsupported final answer. Verification findings: '
                    + json.dumps(completion_verification,ensure_ascii=False)
                )
                if run_budget.used['verification_cycles'] >= run_budget.limits['verification_cycles']:
                    completion_status='incomplete'
                    answer={'role':'assistant','content':candidate_answer + '\n\n(The available evidence did not satisfy every requested requirement; this result is incomplete.)'}
                    break
                messages.append({'role':'system','content':feedback})
                candidate_answer=None
            else:
                raise AgentBudgetExceeded('Agent run stopped: execution iterations budget exhausted (12 iterations used).')
        else:
            # Compatibility path retained for deterministic protocol regression
            # tests and controlled comparison. Normal installations use the
            # planner/executor/verifier architecture above.
            completion_check_pending=False; candidate_answer=None
            completion_check_prompt=(
                "Agent completion check: Reassess the task before deciding to finish. Review the original "
                "user request, the entire in-request conversation, all previous assistant responses, and "
                "the results (including errors) of every tool execution. Determine whether you have obtained "
                "ALL information and completed ALL actions necessary to satisfy the user's actual request "
                "accurately. Do not assume that the task is complete merely because a tool executed "
                "successfully or because you have produced a plausible answer. Identify what information "
                "or actions the request requires, which requirements are already satisfied, and whether any "
                "required information remains missing, incomplete, uncertain, or unverified. Consider whether "
                "any currently available tool can retrieve the missing information or complete an outstanding "
                "action. If more information or action is required and an available tool can provide it, "
                "continue by requesting the appropriate tool call(s) through the normal native structured "
                "tool-calling interface. You may request multiple appropriate calls when useful. If the task "
                "is fully completed, provide the final user-facing answer, grounded in the available results. "
                "If essential information is missing and no available tool can obtain it, ask the user a "
                "concise clarification or explain precisely what remains unavailable. Never invent tool "
                "results or claim that an action was performed unless its result is present in the "
                "conversation. Never describe or print a tool call in prose; request it using native "
                "structured tool_calls."
            )
            for _ in range(16):
                call_messages=messages
                if completion_check_pending:
                    call_messages=list(messages)
                    check_message={'role':'system','content':completion_check_prompt}
                    insert_at=1 if call_messages and call_messages[0].get('role')=='system' else 0
                    call_messages.insert(insert_at,check_message)
                run_budget.consume('execution_iterations')
                run_budget.consume('llm_calls')
                invocation=await model_chat(call_messages,tools,cfg); answer=invocation['message']; llm_index+=1
                usage=invocation.get('usage') or {}
                for key in ('input_tokens','output_tokens'):
                    value=usage.get(key)
                    if isinstance(value,int): usage_totals[key]+=value
                    else: usage_complete=False
                events.append({'type':'llm','index':llm_index,'provider':invocation.get('provider'),'model':invocation.get('model'),'duration_ms':invocation.get('duration_ms'),'usage':usage,'timing':invocation.get('timing') or {},'tool_selection':bool(answer.get('tool_calls')),'completion_check':completion_check_pending,'mcp_servers':mcp_context,'exposed_tool_names':[x['function']['name'] for x in tools],'request':invocation.get('request') or {},'model_response':answer,'response':answer.get('content') or ''})
                reasoning=answer.get('thinking') or answer.get('reasoning_content') or answer.get('reasoning')
                if reasoning and str(cfg.get('reasoning_enabled','false')).lower() in {'1','true','yes','on'}: reasoning_parts.append(str(reasoning))
                calls=answer.get('tool_calls') or []
                if not calls:
                    response_text=answer.get('content')
                    if completion_check_pending:
                        if response_text and str(response_text).strip(): candidate_answer=response_text
                        if candidate_answer and str(candidate_answer).strip(): answer={'role':'assistant','content':candidate_answer}; break
                        raise RuntimeError('The model completion check returned no answer, and no earlier non-empty candidate is available.')
                    if response_text and str(response_text).strip(): candidate_answer=response_text
                    messages.append({'role':'assistant','content':response_text or ''}); completion_check_pending=True; continue
                messages.append({'role':'assistant','content':answer.get('content') or '', 'tool_calls':calls}); candidate_answer=None; completion_check_pending=False
                for call in calls:
                    run_budget.consume('tool_calls')
                    fn=call.get('function',{}) if isinstance(call,dict) else {}; fn=fn if isinstance(fn,dict) else {}; name=fn.get('name',''); raw=fn.get('arguments',{})
                    if isinstance(raw,str):
                        try: raw=json.loads(raw); raw_error=None
                        except json.JSONDecodeError as exc: raw_error=f'Tool arguments are not valid JSON: {exc}'
                    else: raw_error=None
                    call_id=call.get('id','') if isinstance(call,dict) else ''
                    entry={'name':name or 'Invalid native tool call','call_id':call_id,'arguments':raw,'status':'running'}; tool_started=time.perf_counter()
                    try:
                        if not call_id: raise ValueError('The native tool call is missing its required call ID.')
                        if not name: raise ValueError('The native tool call is missing its function name.')
                        if raw_error: raise ValueError(raw_error)
                        if not isinstance(raw,dict): raise ValueError('Tool arguments must be a JSON object.')
                        result=await run_mcp_tool(name,raw); model_result=normalize_mcp_result(result); entry.update(status='success',result=result)
                        messages.append({'role':'tool','tool_call_id':call_id,'name':name,'content':model_result if isinstance(model_result,str) else json.dumps(model_result,ensure_ascii=False)})
                    except Exception as e:
                        entry.update(status='error',result=str(e));
                        if call_id: messages.append({'role':'tool','tool_call_id':call_id,'name':name or 'invalid_tool_call','content':str(e)})
                        else: messages.append({'role':'system','content':'A malformed native tool call was rejected: ' + str(e)})
                    entry['duration_ms']=round((time.perf_counter()-tool_started)*1000); trace.append(entry); events.append({'type':'tool','name':entry['name'],'call_id':call_id,'arguments':entry.get('arguments',raw),'status':entry.get('status'),'result':entry.get('result'),'duration_ms':entry['duration_ms']})
                    run_budget.record_tool_result(entry.get('status') == 'success')
            else: raise AgentBudgetExceeded('Agent run stopped: execution iterations budget exhausted before a final answer was produced.')
        content=answer.get('content') or '(The model returned no text.)'
        usage_totals['complete']=usage_complete; usage_totals['total_tokens']=usage_totals['input_tokens']+usage_totals['output_tokens'] if usage_complete else None
        details=json.dumps({'events':events,'usage':usage_totals,'tools':trace,'reasoning':'\n\n'.join(reasoning_parts),'task_plan':task_plan,'completion_verification':completion_verification,'completion_status':completion_status,'run_budget':run_budget.snapshot()},ensure_ascii=False) if (events or trace or reasoning_parts) else None
        with db() as c: c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(str(uuid.uuid4()),sid,'assistant',content,details,time.time()))
        return {'content':content,'tools':trace,'events':events,'usage':usage_totals,'reasoning':'\n\n'.join(reasoning_parts),'task_plan':task_plan,'completion_verification':completion_verification,'completion_status':completion_status,'run_budget':run_budget.snapshot()}
    except AgentBudgetExceeded as e:
        content=str(e) + ' The task is incomplete; no completion approval was recorded.'
        completion_status='budget_exhausted'
        budget_details={'events':events,'usage':usage_totals,'tools':trace,'reasoning':'\n\n'.join(reasoning_parts),'task_plan':task_plan,'completion_verification':completion_verification,'completion_status':completion_status,'run_budget':run_budget.snapshot()}
        with db() as c: c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(str(uuid.uuid4()),sid,'assistant',content,json.dumps(budget_details,ensure_ascii=False),time.time()))
        return {'content':content,'tools':trace,'events':events,'usage':usage_totals,'reasoning':'\n\n'.join(reasoning_parts),'task_plan':task_plan,'completion_verification':completion_verification,'completion_status':completion_status,'run_budget':run_budget.snapshot()}
    except httpx.HTTPError as e:
        msg=f'Model request failed: {e}'
    except Exception as e: msg=str(e)
    with db() as c: c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(str(uuid.uuid4()),sid,'assistant','Error: '+msg,json.dumps({'events':events,'usage':usage_totals,'tools':trace,'reasoning':'\n\n'.join(reasoning_parts),'task_plan':task_plan,'completion_verification':completion_verification,'completion_status':completion_status,'run_budget':run_budget.snapshot()},ensure_ascii=False),time.time()))
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
