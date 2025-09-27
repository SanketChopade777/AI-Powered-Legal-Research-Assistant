import subprocess
import time
import sys
import os
from threading import Thread


def run_streamlit_app(script_name, port):
    """Run a Streamlit app on a specific port"""
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", script_name, "--server.port", str(port)]
        process = subprocess.Popen(cmd)
        print(f"✅ {script_name} started on http://localhost:{port}")
        return process
    except Exception as e:
        print(f"❌ Error starting {script_name}: {e}")
        return None


def main():
    print("🚀 Starting LegalEase AI Platform...")
    print("=" * 50)

    # Define apps and their ports
    apps = [
        ("main_navigation.py", 8501, "🏠 Navigation Page"),
        ("main.py", 8502, "🤖 Legal Chatbot"),
        ("document_upload.py", 8503, "📄 Document Analysis"),
        ("lawyer_finder.py", 8504, "👨‍💼 Lawyer Finder")
    ]

    processes = []

    # Start all apps
    for script, port, description in apps:
        print(f"Starting {description}...")
        process = run_streamlit_app(script, port)
        if process:
            processes.append(process)
        time.sleep(3)  # Wait between app starts

    print("=" * 50)
    print("🎉 All apps started successfully!")
    print("\n📱 Access Points:")
    print("   • Navigation Page: http://localhost:8501")
    print("   • Legal Chatbot:   http://localhost:8502")
    print("   • Document Analysis: http://localhost:8503")
    print("   • Lawyer Finder:   http://localhost:8504")
    print("\n⏹️  Press Ctrl+C to stop all applications")

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all applications...")
        for process in processes:
            process.terminate()
        print("✅ All applications stopped.")


if __name__ == "__main__":
    main()