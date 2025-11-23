'use client'

import { useState } from 'react'
import { CheckCircle2, XCircle, ChevronRight, Award } from 'lucide-react'

interface QuizQuestion {
  question: string
  options: string[]
  correct: number
  explanation: string
}

interface QuizProps {
  questions: QuizQuestion[]
  onComplete: () => void
  onBack: () => void
}

export default function Quiz({ questions, onComplete, onBack }: QuizProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)
  const [showExplanation, setShowExplanation] = useState(false)
  const [score, setScore] = useState(0)
  const [quizComplete, setQuizComplete] = useState(false)

  const handleAnswer = (index: number) => {
    if (showExplanation) return
    setSelectedAnswer(index)
  }

  const handleSubmit = () => {
    if (selectedAnswer === null) return

    setShowExplanation(true)
    if (selectedAnswer === questions[currentQuestion].correct) {
      setScore(score + 1)
    }
  }

  const handleNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1)
      setSelectedAnswer(null)
      setShowExplanation(false)
    } else {
      setQuizComplete(true)
    }
  }

  const handleRetry = () => {
    setCurrentQuestion(0)
    setSelectedAnswer(null)
    setShowExplanation(false)
    setScore(0)
    setQuizComplete(false)
  }

  if (quizComplete) {
    const percentage = (score / questions.length) * 100
    const passed = percentage >= 70

    return (
      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="text-center space-y-6">
          <div className={`inline-flex items-center justify-center w-24 h-24 rounded-full ${
            passed ? 'bg-green-100' : 'bg-yellow-100'
          }`}>
            {passed ? (
              <Award className="w-16 h-16 text-green-600" />
            ) : (
              <XCircle className="w-16 h-16 text-yellow-600" />
            )}
          </div>

          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              {passed ? 'Congratulations!' : 'Good Try!'}
            </h2>
            <p className="text-gray-600 text-lg">
              You scored {score} out of {questions.length} ({percentage.toFixed(0)}%)
            </p>
          </div>

          {passed ? (
            <div className="bg-green-50 border-2 border-green-200 rounded-lg p-4">
              <p className="text-green-800 font-semibold">
                Excellent work! You've mastered this lesson.
              </p>
            </div>
          ) : (
            <div className="bg-yellow-50 border-2 border-yellow-200 rounded-lg p-4">
              <p className="text-yellow-800 font-semibold">
                You need 70% to pass. Review the material and try again!
              </p>
            </div>
          )}

          <div className="flex items-center justify-center space-x-4">
            {passed ? (
              <button
                onClick={onComplete}
                className="bg-green-600 text-white px-8 py-3 rounded-lg hover:bg-green-700 transition-colors font-semibold"
              >
                Complete Lesson
              </button>
            ) : (
              <>
                <button
                  onClick={onBack}
                  className="bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 transition-colors font-semibold"
                >
                  Review Content
                </button>
                <button
                  onClick={handleRetry}
                  className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors font-semibold"
                >
                  Retry Quiz
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    )
  }

  const question = questions[currentQuestion]
  const isCorrect = selectedAnswer === question.correct

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-6 text-white">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-2xl font-bold">Knowledge Check</h2>
          <span className="text-lg font-semibold">
            Question {currentQuestion + 1} of {questions.length}
          </span>
        </div>
        <div className="w-full bg-indigo-800 rounded-full h-2">
          <div
            className="bg-white h-2 rounded-full transition-all"
            style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
          />
        </div>
      </div>

      <div className="p-8 space-y-6">
        <h3 className="text-2xl font-bold text-gray-900">{question.question}</h3>

        <div className="space-y-3">
          {question.options.map((option, index) => {
            const isSelected = selectedAnswer === index
            const showResult = showExplanation
            const isCorrectAnswer = index === question.correct

            let className = 'p-4 rounded-lg border-2 transition-all cursor-pointer '

            if (!showResult) {
              className += isSelected
                ? 'border-indigo-600 bg-indigo-50'
                : 'border-gray-200 hover:border-indigo-300'
            } else {
              if (isCorrectAnswer) {
                className += 'border-green-600 bg-green-50'
              } else if (isSelected && !isCorrect) {
                className += 'border-red-600 bg-red-50'
              } else {
                className += 'border-gray-200 bg-gray-50'
              }
            }

            return (
              <div
                key={index}
                onClick={() => handleAnswer(index)}
                className={className}
              >
                <div className="flex items-center justify-between">
                  <span className="text-gray-900 font-medium">{option}</span>
                  {showResult && (
                    <>
                      {isCorrectAnswer && (
                        <CheckCircle2 className="w-6 h-6 text-green-600" />
                      )}
                      {isSelected && !isCorrect && (
                        <XCircle className="w-6 h-6 text-red-600" />
                      )}
                    </>
                  )}
                </div>
              </div>
            )
          })}
        </div>

        {showExplanation && (
          <div className={`p-4 rounded-lg border-2 ${
            isCorrect
              ? 'bg-green-50 border-green-200'
              : 'bg-blue-50 border-blue-200'
          }`}>
            <p className={`font-semibold mb-2 ${
              isCorrect ? 'text-green-800' : 'text-blue-800'
            }`}>
              {isCorrect ? '✓ Correct!' : 'ℹ Explanation'}
            </p>
            <p className={isCorrect ? 'text-green-700' : 'text-blue-700'}>
              {question.explanation}
            </p>
          </div>
        )}

        <div className="flex items-center justify-between pt-6 border-t border-gray-200">
          <button
            onClick={onBack}
            className="text-gray-600 hover:text-gray-800 font-semibold"
          >
            Back to Lesson
          </button>

          {!showExplanation ? (
            <button
              onClick={handleSubmit}
              disabled={selectedAnswer === null}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors font-semibold disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Submit Answer
            </button>
          ) : (
            <button
              onClick={handleNext}
              className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors font-semibold flex items-center space-x-2"
            >
              <span>{currentQuestion < questions.length - 1 ? 'Next Question' : 'Finish Quiz'}</span>
              <ChevronRight className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
