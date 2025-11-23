'use client'

import { CheckCircle2, Code, Play } from 'lucide-react'
import { useState } from 'react'

interface Lesson {
  id: string
  title: string
  content: string
  codeExample?: string
  interactiveDemo?: string
  quiz?: any[]
}

interface ModuleContentProps {
  lesson: Lesson
  onComplete: () => void
  onStartQuiz: () => void
  completed: boolean
}

export default function ModuleContent({ lesson, onComplete, onStartQuiz, completed }: ModuleContentProps) {
  const [showCode, setShowCode] = useState(false)
  const [codeOutput, setCodeOutput] = useState('')

  const runCode = () => {
    setCodeOutput('Code execution simulated! In a real environment, this would run the actual code.')
    setTimeout(() => setCodeOutput(''), 3000)
  }

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-8 text-white">
        <h1 className="text-4xl font-bold mb-2">{lesson.title}</h1>
        {completed && (
          <div className="flex items-center space-x-2 text-green-200">
            <CheckCircle2 className="w-5 h-5" />
            <span>Completed</span>
          </div>
        )}
      </div>

      <div className="p-8 space-y-8">
        {/* Content */}
        <div className="prose max-w-none">
          {lesson.content.split('\n\n').map((paragraph, index) => {
            if (paragraph.startsWith('**') && paragraph.endsWith('**')) {
              return (
                <h3 key={index} className="text-2xl font-bold text-gray-900 mt-6 mb-3">
                  {paragraph.replace(/\*\*/g, '')}
                </h3>
              )
            } else if (paragraph.startsWith('- ')) {
              const items = paragraph.split('\n')
              return (
                <ul key={index} className="list-disc pl-6 space-y-2 text-gray-700">
                  {items.map((item, i) => (
                    <li key={i}>{item.replace(/^- /, '').replace(/\*\*/g, '')}</li>
                  ))}
                </ul>
              )
            } else {
              return (
                <p key={index} className="text-gray-700 leading-relaxed mb-4">
                  {paragraph.replace(/\*\*/g, '')}
                </p>
              )
            }
          })}
        </div>

        {/* Code Example */}
        {lesson.codeExample && (
          <div className="bg-gray-50 rounded-lg border-2 border-gray-200 overflow-hidden">
            <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Code className="w-5 h-5 text-green-400" />
                <span className="text-white font-semibold">Code Example</span>
              </div>
              <button
                onClick={() => setShowCode(!showCode)}
                className="text-sm text-indigo-400 hover:text-indigo-300 font-semibold"
              >
                {showCode ? 'Hide' : 'Show'} Code
              </button>
            </div>

            {showCode && (
              <div className="p-4">
                <pre className="bg-gray-900 text-gray-100 p-4 rounded overflow-x-auto text-sm">
                  <code>{lesson.codeExample}</code>
                </pre>

                <div className="mt-4 flex items-center space-x-4">
                  <button
                    onClick={runCode}
                    className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
                  >
                    <Play className="w-4 h-4" />
                    <span>Run Code</span>
                  </button>

                  {codeOutput && (
                    <div className="bg-green-100 text-green-800 px-4 py-2 rounded-lg">
                      {codeOutput}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center space-x-4 pt-6 border-t border-gray-200">
          {lesson.quiz && lesson.quiz.length > 0 && (
            <button
              onClick={onStartQuiz}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors font-semibold"
            >
              Take Quiz
            </button>
          )}

          {!completed && (
            <button
              onClick={onComplete}
              className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors font-semibold flex items-center space-x-2"
            >
              <CheckCircle2 className="w-5 h-5" />
              <span>Mark as Complete</span>
            </button>
          )}

          {completed && (
            <div className="flex items-center space-x-2 text-green-600">
              <CheckCircle2 className="w-6 h-6" />
              <span className="font-semibold">Lesson Completed!</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
