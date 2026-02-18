"use client";
import { useState } from 'react';

export default function Home() {
  const [formData, setFormData] = useState({
    question: '', answer: '', task: '', president: '', date: ''
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
        const response = await fetch('http://localhost:8000/predict/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        const data = await response.json();
        setResult(data.label);
    } catch (error) {
        console.error("Error calling API:", error);
    } finally {
        setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto', padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Clarity</h1>
      <p>
        This app is to decide wether a politician answers clearly on an interview question.
        Please provide a question, answer and - optionaly - the president's name and the date the interview was conducted
      </p>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <div className="flex flex-col gap-1">
          <label className="font-medium">Question</label>
          <input
            className="border p-2 rounded"
            onChange={e => setFormData({ ...formData, question: e.target.value })}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="font-medium">Answer</label>
          <input
            className="border p-2 rounded"
            onChange={e => setFormData({ ...formData, answer: e.target.value })}
          />
        </div>

        <label className="font-medium">Task Type</label>
        <select
          className="border p-2 rounded"
          onChange={e => setFormData({ ...formData, task: e.target.value })}
        >
          <option value="">Select task</option>
          <option value="clarity">Clarity</option>
          <option value="evasion">Evasion</option>
        </select>
        <div className="flex flex-col gap-1">
          <label className="font-medium">President</label>
          <input
            className="border p-2 rounded"
            onChange={e => setFormData({ ...formData, president: e.target.value })}
          />
        </div>
        <input type="date" onChange={e => setFormData({...formData, date: e.target.value})} />
        <button type="submit" disabled={loading} style={{ padding: '10px', cursor: 'pointer' }}>
            {loading ? 'Processing...' : 'Predict'}
        </button>
      </form>
      {result && <h2 style={{ marginTop: '20px', color: 'green' }}>Prediction: {result}</h2>}
    </div>
  );
}