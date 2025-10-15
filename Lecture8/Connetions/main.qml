// main.qml
import QtQuick
import QtQuick.Controls

ApplicationWindow {
    visible: true
    width: 400; height: 200

    Column {
        anchors.centerIn: parent
        spacing: 10

        Text {
            id: label
            text: "Waiting..."
        }

        Button {
            text: "Say Hello"
            onClicked: backend.say_hello()
        }
    }

    Connections {
        target: backend
        function onMessageChanged(msg) {
            label.text = msg
        }
    }
}
