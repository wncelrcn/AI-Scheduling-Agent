"use client";

import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calendar } from "@/components/calendar/Calendar";
import { Card, CardContent } from "@/components/ui/card";
import { useEffect, useState, useRef } from "react";

export default function Dashboard() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState([]);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    // Check if user is logged in, redirect if not
    const storedUsername = sessionStorage.getItem("username");
    if (!storedUsername) {
      router.push("/");
    } else {
      setUsername(storedUsername);
    }
  }, [router]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isChatOpen]);

  const handleLogout = () => {
    sessionStorage.removeItem("username");
    router.push("/");
  };

  const handleSend = async (e) => {
    // prevent default if called from form or enter key
    if (e) e.preventDefault();

    if (!inputValue.trim()) return;

    const userMessage = {
      role: "user",
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    // Optimistic update
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsChatOpen(true); // Open the chat sheet
    setIsProcessing(true);

    // Prepare history for backend (exclude current message which is added manually)
    const history = messages.map(({ role, content }) => ({ role, content }));

    // Simulate AI processing (placeholder for backend call)
    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.content,
          username: username,
          history: history,
        }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();

      const aiMessage = {
        role: "assistant",
        content: data.response,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error("Error processing message:", error);
      // Fallback for demo if backend is not running
      const fallbackMessage = {
        role: "assistant",
        content:
          "I'm having trouble connecting to the scheduling server. (Make sure the backend is running on port 8000). For now, I've noted: " +
          userMessage.content,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, fallbackMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSend(e);
    }
  };

  return (
    <div className="flex flex-col h-screen p-6 bg-background text-foreground overflow-hidden relative">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold">Calendar Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Welcome back, {username}!
          </p>
        </div>
        <div className="flex gap-4">
          <Button variant="outline">Notifications</Button>
          <Button onClick={handleLogout} variant="outline">
            Logout
          </Button>
        </div>
      </div>

      {/* Main Content - Calendar View */}
      <div className="flex-1 flex items-start justify-center overflow-hidden">
        <div className="w-full max-w-7xl h-full max-h-full">
          <Calendar />
        </div>
      </div>

      {/* Chat Interface - Sliding Sheet Overlay */}
      {isChatOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/20 z-40 backdrop-blur-sm transition-opacity duration-300 animate-in fade-in"
            onClick={() => setIsChatOpen(false)}
          />

          {/* Chat Sheet */}
          <Card className="fixed inset-x-0 top-[15vh] h-[70vh] max-h-[600px] z-50 rounded-3xl shadow-2xl border animate-in slide-in-from-bottom duration-500 flex flex-col mx-auto max-w-3xl">
            <div className="px-4 pt-1.5 pb-3 border-b flex justify-between items-center bg-muted/30 rounded-t-3xl">
              <div className="flex flex-col">
                <span className="font-semibold">Scheduling Assistant</span>
                <span className="text-xs text-muted-foreground">
                  AI Agent active
                </span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="rounded-full h-8 w-8 p-0 hover:bg-muted"
                onClick={() => setIsChatOpen(false)}
              >
                ×
              </Button>
            </div>

            <CardContent
              className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth"
              ref={scrollRef}
            >
              {messages.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-50">
                  <p>No messages yet. Start a conversation!</p>
                </div>
              )}
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex flex-col ${
                    msg.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground rounded-tr-sm"
                        : "bg-muted text-foreground rounded-tl-sm"
                    }`}
                  >
                    <div
                      className={`prose prose-sm max-w-none break-words ${
                        msg.role === "user"
                          ? "text-primary-foreground prose-headings:text-primary-foreground prose-p:text-primary-foreground prose-a:text-primary-foreground prose-strong:text-primary-foreground prose-li:text-primary-foreground"
                          : "dark:prose-invert"
                      }`}
                    >
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: ({ node, ...props }) => (
                            <p className="mb-2 last:mb-0" {...props} />
                          ),
                          a: ({ node, ...props }) => (
                            <a
                              className="text-blue-500 hover:underline"
                              target="_blank"
                              rel="noopener noreferrer"
                              {...props}
                            />
                          ),
                          ul: ({ node, ...props }) => (
                            <ul className="list-disc pl-4 mb-2" {...props} />
                          ),
                          ol: ({ node, ...props }) => (
                            <ol className="list-decimal pl-4 mb-2" {...props} />
                          ),
                          li: ({ node, ...props }) => (
                            <li className="mb-1" {...props} />
                          ),
                          code: ({
                            node,
                            inline,
                            className,
                            children,
                            ...props
                          }) => {
                            return inline ? (
                              <code
                                className="bg-black/10 dark:bg-white/10 rounded px-1 py-0.5"
                                {...props}
                              >
                                {children}
                              </code>
                            ) : (
                              <code
                                className="block bg-black/10 dark:bg-white/10 rounded p-2 overflow-x-auto"
                                {...props}
                              >
                                {children}
                              </code>
                            );
                          },
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                  <span className="text-[10px] text-muted-foreground mt-1 px-1">
                    {new Date(msg.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>
              ))}
              {isProcessing && (
                <div className="flex items-start">
                  <div className="bg-muted rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-2">
                    <div
                      className="w-2 h-2 bg-foreground/40 rounded-full animate-bounce"
                      style={{ animationDelay: "0ms" }}
                    />
                    <div
                      className="w-2 h-2 bg-foreground/40 rounded-full animate-bounce"
                      style={{ animationDelay: "150ms" }}
                    />
                    <div
                      className="w-2 h-2 bg-foreground/40 rounded-full animate-bounce"
                      style={{ animationDelay: "300ms" }}
                    />
                  </div>
                </div>
              )}
            </CardContent>

            {/* Chat Input Area inside Sheet */}
            <div className="p-4 border-t bg-background">
              <div className="flex gap-2 max-w-4xl mx-auto">
                <Input
                  placeholder="Type your request here..."
                  className="flex-1"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  autoFocus
                />
                <Button
                  onClick={handleSend}
                  disabled={isProcessing || !inputValue.trim()}
                >
                  Send
                </Button>
              </div>
            </div>
          </Card>
        </>
      )}

      {/* Bottom Input Area (Only visible when chat is closed) */}
      {!isChatOpen && (
        <div className="mt-6 pt-4 border-t">
          <div className="flex gap-2 max-w-4xl mx-auto">
            <Input
              placeholder="Type your request here (e.g., 'Schedule a meeting with John tomorrow at 2pm')..."
              className="flex-1"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <Button
              onClick={handleSend}
              disabled={isProcessing || !inputValue.trim()}
            >
              Send
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
