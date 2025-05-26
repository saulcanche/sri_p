import React, { useState } from 'react';
import './Form.css';

export default function Form({ onSubmit }) {
    const [handle, setHandle] = useState('');
    
    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit(handle);
    };

    return (
        <div className="form-container">
            <form onSubmit={handleSubmit}>
                <input 
                    type="text" 
                    value={handle}
                    onChange={(e) => setHandle(e.target.value)}
                    placeholder="Ingresa tu handle de Codeforces"
                    required
                />
                <button type="submit">Obtener Recomendaciones</button>
            </form>
        </div>
    );
}