// main.qml
import QtQuick
import QtQuick.Controls

ApplicationWindow {
    visible: true
    width: 400
    height: 300
    title: "Hello QML"

    Button {
        text: "Click Me"
        anchors.centerIn: parent
    }
}
