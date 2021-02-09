const React = require('react');
const ReactDOM = require('react-dom');

function App() {
    return (
        <div>
        <h1>Hi From Javascript Land!</h1>
        </div>
    );
}

export default App;

ReactDOM.render(
    <App />,
    document.getElementById('react')
)
