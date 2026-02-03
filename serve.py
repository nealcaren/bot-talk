# /// script
# dependencies = ["fastapi==0.115.0", "uvicorn==0.30.6", "jinja2==3.1.4"]
# ///

import os
import uvicorn

from main import app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
