import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BuiltinPromptMigrationTests(unittest.TestCase):
    def test_legacy_default_prompt_rows_are_removed_once_and_not_recreated(self):
        with tempfile.TemporaryDirectory() as data_dir:
            env = dict(os.environ, AGENT_PRO_HOME=data_dir, PYTHONPATH=str(ROOT))
            script = r"""
import time, uuid
import app.main as app
with app.db() as c:
    c.execute("DELETE FROM settings WHERE key='builtin_prompt_cleanup_v1'")
    for content in ('old default wording', app.DEFAULT_SYSTEM):
        c.execute('INSERT INTO prompts(id,name,description,content,enabled,created) VALUES(?,?,?,?,1,?)',
                  (str(uuid.uuid4()), 'Default agent instructions', '', content, time.time()))
app.init_db()
with app.db() as c:
    assert c.execute("SELECT COUNT(*) FROM prompts WHERE name='Default agent instructions'").fetchone()[0] == 0
    assert c.execute("SELECT value FROM settings WHERE key='system_prompt'").fetchone()[0] == app.DEFAULT_SYSTEM
app.init_db()
with app.db() as c:
    assert c.execute("SELECT COUNT(*) FROM prompts WHERE name='Default agent instructions'").fetchone()[0] == 0
"""
            result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
