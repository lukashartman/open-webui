import os
import tempfile

os.environ.setdefault('WEBUI_SECRET_KEY', 'test-secret-key-for-unit-tests')
os.environ.setdefault('DATA_DIR', tempfile.mkdtemp(prefix='open-webui-test-'))
