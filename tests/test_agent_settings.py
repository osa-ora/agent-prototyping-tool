import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AgentSettingsTests(unittest.TestCase):
    def run_isolated(self, script):
        with tempfile.TemporaryDirectory() as data_dir:
            env = dict(os.environ, AGENT_PRO_HOME=data_dir, PYTHONPATH=str(ROOT))
            result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_agent_limits_are_seeded_and_saved_for_future_runs(self):
        self.run_isolated(r'''
import asyncio
import app.main as app
cfg=app.setting_map()
expected={
 'agent_architecture':'true', 'agent_max_llm_calls':'30',
 'agent_max_execution_iterations':'12', 'agent_max_tool_calls':'20',
 'agent_max_verification_cycles':'2', 'agent_max_planning_attempts':'2',
 'agent_max_verifier_attempts':'4', 'agent_verifier_attempts_per_cycle':'2',
 'agent_max_consecutive_tool_errors':'4'}
for key,value in expected.items(): assert cfg[key]==value, (key,cfg.get(key))
asyncio.run(app.save_settings(app.Obj(data={
 'agent_max_llm_calls':48, 'agent_max_planning_attempts':3,
 'agent_max_verification_cycles':5, 'agent_verifier_attempts_per_cycle':3,
 'agent_architecture':False})))
cfg=app.setting_map()
assert cfg['agent_max_llm_calls']=='48'
assert cfg['agent_max_planning_attempts']=='3'
assert cfg['agent_max_verification_cycles']=='5'
assert cfg['agent_verifier_attempts_per_cycle']=='3'
assert cfg['agent_architecture']=='false'
''')

    def test_agent_limits_reject_out_of_range_values(self):
        self.run_isolated(r'''
import asyncio
from fastapi import HTTPException
import app.main as app
for data in ({'agent_max_llm_calls':0}, {'agent_max_tool_calls':201},
             {'agent_verifier_attempts_per_cycle':11}, {'agent_architecture':'maybe'}):
    try: asyncio.run(app.save_settings(app.Obj(data=data)))
    except HTTPException as exc: assert exc.status_code==400
    else: raise AssertionError('Invalid setting accepted: '+repr(data))
''')

    def test_agent_limits_are_used_when_constructing_run_budget(self):
        self.run_isolated(r'''
import app.main as app
from app.agent_engine import AgentRunBudget
cfg=app.setting_map()
run_budget=AgentRunBudget(
 max_llm_calls=int(cfg.get('agent_max_llm_calls',30)),
 max_execution_iterations=int(cfg.get('agent_max_execution_iterations',12)),
 max_tool_calls=int(cfg.get('agent_max_tool_calls',20)),
 max_verification_cycles=int(cfg.get('agent_max_verification_cycles',2)),
 max_verifier_attempts=int(cfg.get('agent_max_verifier_attempts',4)),
 max_consecutive_tool_errors=int(cfg.get('agent_max_consecutive_tool_errors',4)))
assert run_budget.limits['llm_calls']==30
assert run_budget.limits['execution_iterations']==12
assert run_budget.limits['tool_calls']==20
assert run_budget.limits['verification_cycles']==2
assert run_budget.limits['verifier_attempts']==4
assert run_budget.limits['consecutive_tool_errors']==4
''')


if __name__ == '__main__':
    unittest.main()
