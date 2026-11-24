/**
 * App.tsx — Antarmuka Chatbot Jaringan Komputer
 * Menggunakan Tailwind CSS utility-first classes secara langsung.
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, AlertCircle, MessageSquareText, Sparkles, BrainCircuit, ChevronDown, X, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { sendChatMessage } from './services/api';
import type { ChatResponse, ChatMode } from './services/api';

interface Message {
  id: string;
  type: 'user' | 'bot' | 'error';
  content: string;
  retrieval?: ChatResponse['retrieval'];
  latency_ms?: number;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [selectedMode, setSelectedMode] = useState<ChatMode>('berpikir');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSelectSuggestion = (suggestion: string) => {
    setInput(suggestion);
    inputRef.current?.focus();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await sendChatMessage(userMessage.content, selectedMode);

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response.answer,
        retrieval: response.retrieval,
        latency_ms: response.latency_ms,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error: any) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'error',
        content: error.message || 'Terjadi kesalahan saat menghubungi server.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  return (
    <div className="flex min-h-dvh w-full bg-[#FFFBE6] text-[#17301F] font-sans antialiased">
      <main className="flex-1 flex flex-col h-dvh relative overflow-hidden">
        {/* Header */}
        <header className="h-[60px] px-5 flex items-center justify-start gap-3 border-b border-[#DDE6D6] bg-[#FFFBE6]/90 sticky top-0 z-20 backdrop-blur-md">
          <img
            src="/logo-chatbot.png"
            alt="Logo Chatbot"
            className="w-9 h-9 object-contain"
          />
          <div>
            <h1 className="text-lg font-bold text-[#17301F] leading-tight">Chatbot Jaringan Komputer</h1>
            <p className="text-[13px] text-[#5F6F64] leading-tight">Asisten Belajar Mahasiswa Teknik Informatika</p>
          </div>
        </header>

        {/* Area Percakapan */}
        <section className="flex-1 overflow-y-auto scroll-smooth">
          <div className="w-full max-w-[860px] mx-auto px-4 sm:px-6 pt-8 pb-32 flex flex-col gap-6">
            {messages.length === 0 ? (
              /* Tampilan Kosong (Empty State) */
              <div className="flex flex-col items-center justify-center text-center gap-4 mt-[10vh]">
                <div className="w-14 h-14 grid place-items-center bg-[#D5ED9F] text-[#00712D] rounded-[18px] shadow-xs">
                  <Bot size={32} />
                </div>
                <div className="text-2xl sm:text-3xl font-bold tracking-tight text-[#17301F] leading-tight">
                  Halo! Saya asisten<br />Jaringan Komputer Anda.
                </div>
                <div className="text-sm text-[#5F6F64] max-w-md">
                  Saya dapat membantu Anda memahami materi kuliah berdasarkan dokumen yang telah dipelajari.
                </div>

                <div className="flex flex-wrap justify-center gap-2.5 mt-4">
                  <button
                    type="button"
                    onClick={() => handleSelectSuggestion("Apa itu model OSI dan sebutkan lapisannya?")}
                    className="bg-[#FFFFF8] border border-[#DDE6D6] px-4 py-2 rounded-full text-[#5F6F64] text-sm hover:bg-[#EEF7D8] hover:text-[#00712D] hover:border-[#D5ED9F] transition-all cursor-pointer shadow-xs"
                  >
                    Apa itu model OSI dan sebutkan lapisannya?
                  </button>
                  <button
                    type="button"
                    onClick={() => handleSelectSuggestion("Jelaskan cara kerja subnet mask pada IPv4.")}
                    className="bg-[#FFFFF8] border border-[#DDE6D6] px-4 py-2 rounded-full text-[#5F6F64] text-sm hover:bg-[#EEF7D8] hover:text-[#00712D] hover:border-[#D5ED9F] transition-all cursor-pointer shadow-xs"
                  >
                    Jelaskan cara kerja subnet mask pada IPv4.
                  </button>
                  <button
                    type="button"
                    onClick={() => handleSelectSuggestion("Apa perbedaan antara routing statis dan dinamis?")}
                    className="bg-[#FFFFF8] border border-[#DDE6D6] px-4 py-2 rounded-full text-[#5F6F64] text-sm hover:bg-[#EEF7D8] hover:text-[#00712D] hover:border-[#D5ED9F] transition-all cursor-pointer shadow-xs"
                  >
                    Apa perbedaan antara routing statis dan dinamis?
                  </button>
                </div>
              </div>
            ) : (
              <>
                {/* Daftar Pesan */}
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex w-full ${msg.type === 'user' ? 'justify-end' : 'justify-start gap-3'}`}
                  >
                    {msg.type !== 'user' && (
                      <div
                        className={`w-8 h-8 shrink-0 grid place-items-center rounded-[10px] shadow-xs text-white ${
                          msg.type === 'error' ? 'bg-[#B42318]' : 'bg-[#00712D]'
                        }`}
                      >
                        {msg.type === 'error' ? <AlertCircle size={20} /> : <Bot size={20} />}
                      </div>
                    )}

                    <div
                      className={
                        msg.type === 'user'
                          ? 'max-w-[85%] sm:max-w-[640px] px-4 py-3 bg-[#D5ED9F] text-[#17301F] rounded-[18px_18px_6px_18px] text-[15.5px] leading-relaxed shadow-xs'
                          : `max-w-[760px] text-[15.5px] leading-relaxed pt-1 markdown-prose ${
                              msg.type === 'error' ? 'text-[#B42318]' : 'text-[#17301F]'
                            }`
                      }
                    >
                      {msg.type === 'user' ? (
                        msg.content
                      ) : (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content || ''}</ReactMarkdown>
                      )}

                      {/* Sumber Dokumen RAG */}
                      {msg.retrieval && msg.retrieval.sources && msg.retrieval.sources.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-[#DDE6D6]">
                          <div className="flex items-center gap-1.5 text-[13px] font-semibold text-[#5F6F64] mb-3">
                            <span>Sumber Konteks</span>
                          </div>
                          <ul className="flex flex-col gap-2 list-none p-0 m-0">
                            {msg.retrieval.sources.map((src, idx) => (
                              <li
                                key={idx}
                                className="p-3.5 bg-[#FFFFF8] border border-[#DDE6D6] rounded-2xl hover:border-[#BFD1B5] hover:shadow-xs transition-all text-[13.5px] text-[#5F6F64] flex justify-between items-center"
                              >
                                <span>📄 {src.source} (Hal. {src.page})</span>
                                {src.score !== undefined && src.score !== null && (
                                  <span className="inline-flex items-center px-2 py-1 rounded-full bg-[#D5ED9F] text-[#00712D] text-xs font-bold ml-3 shrink-0">
                                    {src.score.toFixed(3)}
                                  </span>
                                )}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {/* Indikator Loading */}
                {isLoading && (
                  <div className="flex w-full justify-start gap-3">
                    <div className="w-8 h-8 shrink-0 grid place-items-center rounded-[10px] shadow-xs text-white bg-[#00712D]">
                      <Bot size={20} />
                    </div>
                    <div className="max-w-[760px] pt-2">
                      <div className="flex items-center gap-1.5 py-1">
                        <span className="w-2 h-2 bg-[#8A968D] rounded-full animate-typing-1"></span>
                        <span className="w-2 h-2 bg-[#8A968D] rounded-full animate-typing-2"></span>
                        <span className="w-2 h-2 bg-[#8A968D] rounded-full animate-typing-3"></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </div>
        </section>

        {/* Input Bar / Composer */}
        <footer className="sticky bottom-0 w-full bg-gradient-to-t from-[#FFFBE6] via-[#FFFBE6]/95 to-transparent backdrop-blur-md px-3 sm:px-5 pt-3 pb-5 z-20">
          <div className="w-full max-w-[860px] mx-auto flex flex-col gap-2">
            <form
              onSubmit={handleSubmit}
              className="flex items-center gap-2 min-h-[52px] sm:min-h-[56px] px-2.5 sm:px-3 py-1.5 bg-[#FFFFF8] border-[1.5px] border-[#BFD1B5] rounded-2xl sm:rounded-3xl shadow-sm transition-all focus-within:border-[#00712D] focus-within:bg-white focus-within:ring-4 focus-within:ring-[#00712D]/15"
            >
              {/* Selector Mode */}
              <div
                className="relative inline-flex items-center gap-1.5 h-[34px] sm:h-[38px] pl-2 sm:pl-2.5 pr-6 sm:pr-7 bg-[#F4F7E8] border border-[#DDE6D6] rounded-full hover:bg-[#EEF7D8] hover:border-[#BFD1B5] focus-within:border-[#00712D] focus-within:ring-2 focus-within:ring-[#00712D]/20 transition-all shrink-0 cursor-pointer"
                title={
                  selectedMode === 'cepat'
                    ? 'Mode Cepat: Jawaban ringkas dan cepat'
                    : 'Mode Berpikir: Jawaban komprehensif dengan analisis RAG'
                }
              >
                <div className="flex items-center justify-center pointer-events-none shrink-0" aria-hidden="true">
                  {selectedMode === 'cepat' ? (
                    <Sparkles size={16} className="text-amber-600" />
                  ) : (
                    <BrainCircuit size={16} className="text-[#00712D]" />
                  )}
                </div>
                <select
                  value={selectedMode}
                  onChange={(e) => setSelectedMode(e.target.value as ChatMode)}
                  className="appearance-none bg-transparent border-none text-[#17301F] text-[12.5px] sm:text-[13.5px] font-semibold cursor-pointer outline-none pr-1 disabled:cursor-not-allowed disabled:opacity-60"
                  aria-label="Pilih mode jawaban"
                  disabled={isLoading}
                >
                  <option value="cepat">Mode Cepat</option>
                  <option value="berpikir">Mode Berpikir</option>
                </select>
                <ChevronDown size={14} className="absolute right-2 text-[#5F6F64] pointer-events-none" aria-hidden="true" />
              </div>

              {/* Divider */}
              <div className="w-px h-[18px] sm:h-[24px] bg-[#DDE6D6] shrink-0 mx-0.5" aria-hidden="true" />

              {/* Input Wrap */}
              <div className="flex items-center gap-2 sm:gap-2.5 flex-1 min-w-0">
                <div className="hidden sm:flex items-center justify-center text-[#8A968D] shrink-0" aria-hidden="true">
                  <MessageSquareText size={18} />
                </div>
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') {
                      setInput('');
                    }
                  }}
                  placeholder={
                    selectedMode === 'cepat'
                      ? 'Tanyakan materi Jaringan Komputer (Mode Cepat)...'
                      : 'Tanyakan materi Jaringan Komputer (Mode Berpikir)...'
                  }
                  disabled={isLoading}
                  className="flex-1 min-w-0 h-[38px] sm:h-[42px] border-none outline-none bg-transparent text-[#17301F] text-sm sm:text-[15px] p-0 placeholder:text-[#8A968D] placeholder:text-[13px] sm:placeholder:text-[14.5px]"
                />
                {input.trim() && !isLoading && (
                  <button
                    type="button"
                    onClick={() => {
                      setInput('');
                      inputRef.current?.focus();
                    }}
                    className="flex items-center justify-center w-[26px] h-[26px] rounded-full bg-[#F4F7E8] text-[#5F6F64] hover:bg-[#DDE6D6] hover:text-[#17301F] transition-all shrink-0 cursor-pointer"
                    aria-label="Hapus teks"
                    title="Hapus teks pertanyaan"
                  >
                    <X size={15} />
                  </button>
                )}
                <button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  className="flex items-center justify-center w-9 h-9 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-[#00712D] to-[#005C24] text-white shrink-0 shadow-md shadow-[#00712D]/25 transition-all hover:not-disabled:-translate-y-0.5 hover:not-disabled:shadow-lg hover:not-disabled:shadow-[#00712D]/35 active:not-disabled:translate-y-0 disabled:bg-[#F4F7E8] disabled:text-[#8A968D] disabled:border disabled:border-[#DDE6D6] disabled:shadow-none disabled:cursor-not-allowed disabled:opacity-60 cursor-pointer"
                  aria-label="Kirim pesan"
                  title={
                    isLoading
                      ? 'Sedang memproses jawaban...'
                      : input.trim()
                      ? 'Kirim pesan'
                      : 'Tulis pertanyaan terlebih dahulu'
                  }
                >
                  {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={17} />}
                </button>
              </div>
            </form>
            <p className="text-center text-xs text-[#8A968D] leading-relaxed select-none m-0">
              Chatbot dapat membuat kesalahan. Jadikan sebagai pendamping belajar, bukan sumber tunggal.
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;
