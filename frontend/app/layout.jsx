export const metadata = {
  title: 'Clarity LLM',
  description: 'Model Prediction Interface',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}