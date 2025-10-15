#!/usr/bin/env -S uv run --script

import sys

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication


class Backend(QObject):
    messageChanged = Signal(str)

    @Slot()
    def say_hello(self):
        self.messageChanged.emit("Hello from Python!")


# main.py

app = QApplication(sys.argv)
engine = QQmlApplicationEngine()

# Create the backend and set it as a context property BEFORE loading the QML
backend = Backend()
engine.rootContext().setContextProperty("backend", backend)

engine.load("main.qml")

if not engine.rootObjects():
    sys.exit(-1)

sys.exit(app.exec())
