class Chatbox {
    constructor() {
        this.args = {
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const { chatBox, sendButton } = this.args;

        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hide the box
        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value;
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 };
        this.messages = [];
        this.messages.push(msg1);


        fetch('/chatbot', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(r => {
                let msg2 = { name: "Destroyer", message: r.answer };
                this.messages.push(msg2);
                this.updateChatText(chatbox);
                textField.value = '';

            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox);
                textField.value = '';
            });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().forEach(function (item, index) {
            if (item.name === "Destroyer") {
                html += '<div class="messages_item messages_item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages_item messages_item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.insertAdjacentHTML('beforeend', html);
    }
}

const chatbox = new Chatbox();
chatbox.display(); 
chatbox.toggleState(chatbox.args.chatBox);

document.addEventListener('DOMContentLoaded', function() {
    document.querySelector('.chatbox_support').classList.add('chatbox_active');
});