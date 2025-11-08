'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

export default function AIBot() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string>('');
    const [isConnected, setIsConnected] = useState<boolean>(true);
    const inputRef = useRef<HTMLInputElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        inputRef.current?.focus();
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const sendMessage = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input.trim(),
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMessage.content,
                    session_id: sessionId || undefined,
                }),
            });

            if (!response.ok) throw new Error('Network error');
            const data = await response.json();

            if (!sessionId) setSessionId(data.session_id);

            const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: data.response,
                timestamp: new Date(),
            };

            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            console.error('Error:', error);
            setIsConnected(false);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: '⚠️ I encountered an issue connecting to the server. Please try again shortly.',
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="flex flex-col h-full bg-white rounded-2xl shadow-2xl border border-gray-100 overflow-hidden">
            {/* HEADER */}
            <div className="bg-gradient-to-r from-teal-600 to-emerald-600 text-white p-5 flex items-center justify-between rounded-t-2xl shadow-md">
                <div className="flex items-center gap-3">
                    <Bot className="w-7 h-7" />
                    <h2 className="text-2xl font-semibold tracking-tight">MedExtract AI Intelligence Assistant</h2>
                </div>
                <div className="flex items-center gap-2 text-sm text-teal-100">
                    {isConnected ? (
                        <>
                            <div className="w-2 h-2 bg-emerald-300 rounded-full animate-pulse"></div>
                            <span>Online</span>
                        </>
                    ) : (
                        <>
                            <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                            <span>Offline</span>
                        </>
                    )}
                </div>
            </div>

            {/* CHAT BODY */}
            <div className="flex-1 overflow-y-auto px-6 py-6 bg-gray-50 space-y-5 relative">
                {messages.length === 0 && (
                    <div className="text-center text-gray-500 mt-20">
                        <Sparkles className="w-14 h-14 mx-auto mb-4 text-teal-400" />
                        <p className="text-lg font-semibold">Hello there 👋</p>
                        <p className="text-gray-600 mt-1">Ask me anything about cancer — from types, symptoms, and treatments to risk factors, prevention, and recent research findings</p>
                    </div>
                )}

                <AnimatePresence>
                    {messages.map((msg) => (
                        <motion.div
                            key={msg.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.2 }}
                            className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            {msg.role === 'assistant' && (
                                <div className="flex-shrink-0">
                                    <div className="w-9 h-9 bg-teal-500 rounded-full flex items-center justify-center shadow-md">
                                        <Bot className="w-5 h-5 text-white" />
                                    </div>
                                </div>
                            )}

                            <motion.div
                                whileHover={{ scale: 1.02 }}
                                className={`max-w-[75%] rounded-2xl p-4 text-sm leading-relaxed ${
                                    msg.role === 'user'
                                        ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-br-none shadow-lg'
                                        : 'bg-white border border-gray-200 text-gray-800 rounded-tl-none shadow-md'
                                }`}
                            >
                                <p className="whitespace-pre-wrap">{msg.content}</p>
                                <p
                                    className={`text-xs mt-2 text-right ${
                                        msg.role === 'user' ? 'text-emerald-200' : 'text-gray-500'
                                    }`}
                                >
                                    {msg.timestamp.toLocaleTimeString([], {
                                        hour: '2-digit',
                                        minute: '2-digit',
                                    })}
                                </p>
                            </motion.div>

                            {msg.role === 'user' && (
                                <div className="flex-shrink-0">
                                    <div className="w-9 h-9 bg-gray-600 rounded-full flex items-center justify-center shadow-md">
                                        <User className="w-5 h-5 text-white" />
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    ))}
                </AnimatePresence>

                {/* Typing Indicator */}
                {isLoading && (
                    <div className="flex gap-3 justify-start">
                        <div className="w-9 h-9 bg-teal-500 rounded-full flex items-center justify-center shadow-md">
                            <Bot className="w-5 h-5 text-white" />
                        </div>
                        <div className="bg-white border border-gray-200 rounded-2xl p-4 rounded-tl-none shadow-md">
                            <div className="flex space-x-2 items-center">
                                <div className="w-2.5 h-2.5 bg-teal-400 rounded-full animate-bounce delay-75" />
                                <div className="w-2.5 h-2.5 bg-teal-400 rounded-full animate-bounce delay-150" />
                                <div className="w-2.5 h-2.5 bg-teal-400 rounded-full animate-bounce delay-300" />
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* INPUT BAR */}
            <div className="border-t border-gray-200 p-4 bg-white rounded-b-2xl">
                <div className="flex gap-3 items-center">
                    <input
                        ref={inputRef}
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyPress}
                        placeholder="........"
                        className="flex-1 px-5 py-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-teal-300 text-gray-800 transition-all"
                        disabled={isLoading}
                    />
                    <button
                        onClick={sendMessage}
                        disabled={!input.trim() || isLoading}
                        className="p-3 bg-gradient-to-r from-teal-500 to-emerald-600 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-60"
                    >
                        {isLoading ? <Loader2 className="w-6 h-6 animate-spin" /> : <Send className="w-6 h-6" />}
                    </button>
                </div>
            </div>
        </div>
    );
}
