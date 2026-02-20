'use client';

import { useState } from 'react';
import { Container, Form, Button, Card, Spinner, Row, Col } from 'react-bootstrap';

export default function Home() {
  const [formData, setFormData] = useState({
    question: '',
    answer: '',
    president: '',
    date: '',
    task: 'clarity'
  });
  const [llmResponse, setLlmResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setLlmResponse('');

    try {
      // Replace with your actual backend endpoint
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      
      const data = await res.json();
      setLlmResponse(data.result); 
    } catch (error) {
      console.error("Error fetching LLM response:", error);
      setLlmResponse("There was an error connecting to the backend.");
    } finally {
      setIsLoading(false);
    }
  };

  // Color Palette
  const deepBlue = "#0B1B3D";
  const containerBlue = "#1C3A6B";
  const fieldBlue = "#8ca2c0"; // Light enough for black text to be readable

  return (
    <div style={{ backgroundColor: deepBlue, minHeight: '100vh', display: 'flex', alignItems: 'center', padding: '20px' }}>
      <Container style={{ maxWidth: '800px', backgroundColor: containerBlue, padding: '40px', borderRadius: '15px', color: '#ffffff', boxShadow: '0 10px 30px rgba(0,0,0,0.5)' }}>
        <h2 className="mb-4 text-center fw-bold">Clarity - Political Evasion Detection</h2>
        <p className="text-left"> Provide a question and answer given by a politician and (optionally) the politician's name and the date the question was conducted. </p>
        

        <p className="text-left"> Clarity will then decide if his/her answer was clear. </p>

        <Form onSubmit={handleSubmit}>
          
          <Form.Group className="mb-3 d-flex flex-column text-start">
            <Form.Label className="fw-semibold mb-1">Question</Form.Label>
            <Form.Control 
              as="textarea" 
              rows={2} 
              name="question" 
              onChange={handleChange} 
              required 
              style={{ backgroundColor: fieldBlue, border: 'none' }}
            />
          </Form.Group>

          <Form.Group className="mb-3 d-flex flex-column text-start">
            <Form.Label className="fw-semibold mb-1">Answer</Form.Label>
            <Form.Control 
              as="textarea" 
              rows={3} 
              name="answer" 
              onChange={handleChange} 
              required 
              style={{ backgroundColor: fieldBlue, border: 'none' }}
            />
          </Form.Group>
          
          
          
          
          <Row>
            <Col md={6}>
              <Form.Group className="mb-3 d-flex flex-column text-start">
                <Form.Label className="fw-semibold mb-1">President <span className="text-muted fw-normal" style={{fontSize: '0.85em'}}>(Optional)</span></Form.Label>
                <Form.Control 
                  type="text" 
                  name="president" 
                  placeholder="e.g., Abraham Lincoln" 
                  value={formData.president}
                  onChange={handleChange} 
                  /* 'required' attribute removed here */
                  style={{ backgroundColor: fieldBlue, border: 'none' }}
                />
              </Form.Group>
            </Col>
            <Col md={6}>
               <Form.Group className="mb-3 d-flex flex-column text-start">
                <Form.Label className="fw-semibold mb-1">Date <span className="text-muted fw-normal" style={{fontSize: '0.85em'}}>(Optional)</span></Form.Label>
                <Form.Control 
                  type="date" 
                  name="date" 
                  value={formData.date}
                  onChange={handleChange} 
                  /* 'required' attribute removed here */
                  style={{ backgroundColor: fieldBlue, border: 'none' }}
                />
              </Form.Group>
            </Col>
          </Row>

          

          <Form.Group className="mb-4 d-flex flex-column text-start">
            <Form.Label className="fw-semibold mb-1">Task</Form.Label>
            <Form.Select 
              name="task" 
              onChange={handleChange}
              style={{ backgroundColor: fieldBlue, border: 'none' }}
            >
              <option value="clarity">Clarity Analysis</option>
              <option value="evasion">Evasion Detection</option>
            </Form.Select>
          </Form.Group>

          <Button variant="light" type="submit" disabled={isLoading} className="w-100 fw-bold fs-5 py-2 mt-2" style={{ color: containerBlue }}>
            {isLoading ? (
              <><Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" className="me-2"/> Processing...</>
            ) : (
              'Clarify'
            )}
          </Button>
        </Form>

        {/* Response Card */}
        {llmResponse && (
          <Card className="mt-4 border-0" style={{ backgroundColor: fieldBlue, color: '#333' }}>
            <Card.Header className="fw-bold" style={{ backgroundColor: '#ffffff', borderBottom: 'none' }}>Analysis Result</Card.Header>
            <Card.Body>
              <Card.Text style={{ whiteSpace: 'pre-wrap' }}>
                {llmResponse}
              </Card.Text>
            </Card.Body>
          </Card>
        )}
      </Container>
    </div>
  );
}