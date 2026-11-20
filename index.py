"""
Vercel Web Entrypoint
Exports the Flask application for Vercel Serverless Functions.
"""
from app_web import app

# Export application for WSGI
application = app

if __name__ == "__main__":
    app.run()
