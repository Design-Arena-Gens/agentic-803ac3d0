import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Generative AI: Zero to Hero',
  description: 'Interactive course to master Generative AI from fundamentals to advanced applications',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50">{children}</body>
    </html>
  )
}
