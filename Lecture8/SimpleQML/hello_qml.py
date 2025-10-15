#!/usr/bin/env -S uv run --script
import sys

# QQmlApplicationEngine provides a way to load and run QML (Qt Modeling Language) files.
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

# Create a QML engine. This engine is responsible for loading QML files
# and creating the object trees defined within them.
engine = QQmlApplicationEngine()

# Load the user interface from the 'hello.qml' file.
# The QML file defines the structure, appearance, and behavior of the UI.
# The engine parses this file and instantiates the QML objects.
engine.load("hello.qml")


# This is a crucial check to ensure the QML file was loaded successfully.
# If the engine.rootObjects() list is empty, it means that no QML objects
# could be created. This could be due to errors in the QML file (e.g., syntax errors)
# or if the file doesn't exist.
if not engine.rootObjects():
    # If loading fails, the program exits with a non-zero status code,
    # which is a standard way to indicate that an error occurred.
    sys.exit(-1)

# This starts the application's event loop. The app.exec() call enters the main
# loop and waits for user interaction (like mouse clicks or key presses) until
# the application is closed. The script will block here until the user closes
# the main window.
# The return value of app.exec() is the application's exit status, which is
# then passed to sys.exit() to ensure a clean shutdown.
sys.exit(app.exec())
