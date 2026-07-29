"""Start Merger Studio WebUI server."""
import webbrowser, time
from WebUI.page_trainer import start_webui_server

server = start_webui_server(port=6789, model_dir='workspace/model')
webbrowser.open('http://localhost:6789/MergerStudio')
print('Merger Studio: http://localhost:6789/MergerStudio')
print('Press Ctrl+C to stop.')
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.shutdown()
