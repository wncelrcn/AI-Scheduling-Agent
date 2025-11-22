"use client";

import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { createClient } from "@/utils/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Calendar } from "@/components/calendar/Calendar";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useEffect, useState, useRef } from "react";

export default function Dashboard() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState([]);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const scrollRef = useRef(null);

  // Settings State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [workStart, setWorkStart] = useState("09:00");
  const [workEnd, setWorkEnd] = useState("17:00");
  const [workingDays, setWorkingDays] = useState([
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
  ]);
  const [isSaving, setIsSaving] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);

  // Participants State
  const [availableUsers, setAvailableUsers] = useState([]);
  const [selectedParticipants, setSelectedParticipants] = useState([]);
  const [isParticipantsOpen, setIsParticipantsOpen] = useState(false);

  const daysOfWeek = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  const supabase = createClient();

  useEffect(() => {
    // Check if user is logged in, redirect if not
    const storedUsername = sessionStorage.getItem("username");
    if (!storedUsername) {
      router.push("/");
    } else {
      setUsername(storedUsername);
      fetchUserPreferences(storedUsername);
      fetchAvailableUsers(storedUsername);
    }
  }, [router]);

  const fetchAvailableUsers = async (currentUser) => {
    try {
      const { data, error } = await supabase
        .from("users")
        .select("name, work_start, work_end, working_days")
        .neq("name", currentUser); // Don't show self in list

      if (data) {
        setAvailableUsers(data);
      }
    } catch (err) {
      console.error("Error fetching users:", err);
    }
  };

  const fetchUserPreferences = async (user) => {
    try {
      const { data, error } = await supabase
        .from("users")
        .select("work_start, work_end, working_days")
        .eq("name", user)
        .single();

      if (data) {
        if (data.work_start) {
          // Parse timetz string (e.g., "09:00:00+08") back to HH:MM
          const timePart = data.work_start.split("+")[0].split("-")[0]; // Handle + or - offset
          const [h, m] = timePart.split(":");
          setWorkStart(`${h}:${m}`);
        }
        if (data.work_end) {
          const timePart = data.work_end.split("+")[0].split("-")[0];
          const [h, m] = timePart.split(":");
          setWorkEnd(`${h}:${m}`);
        }
        if (data.working_days) {
          setWorkingDays(data.working_days);
        }
      }
    } catch (err) {
      console.error("Error fetching preferences:", err);
    }
  };

  const handleSavePreferences = async () => {
    setIsSaving(true);
    try {
      // For timetz, we just need the time string + timezone offset
      // Construct ISO-like time string with offset (e.g., "09:00:00+08")

      const getTimeWithOffset = (timeStr) => {
        // Force UTC+8 (Singapore Time) as requested
        return `${timeStr}:00+08:00`;
      };

      const { error } = await supabase.from("users").upsert({
        name: username,
        work_start: getTimeWithOffset(workStart),
        work_end: getTimeWithOffset(workEnd),
        working_days: workingDays,
      });

      if (error) throw error;
      setIsSettingsOpen(false);
      setShowSuccessModal(true);
    } catch (err) {
      console.error("Error saving preferences:", err);
      alert("Failed to save preferences");
    } finally {
      setIsSaving(false);
    }
  };

  const toggleDay = (day) => {
    if (workingDays.includes(day)) {
      setWorkingDays(workingDays.filter((d) => d !== day));
    } else {
      // Sort days to keep them in order
      const newDays = [...workingDays, day];
      newDays.sort((a, b) => daysOfWeek.indexOf(a) - daysOfWeek.indexOf(b));
      setWorkingDays(newDays);
    }
  };

  const toggleParticipant = (userName) => {
    if (selectedParticipants.includes(userName)) {
      setSelectedParticipants(
        selectedParticipants.filter((p) => p !== userName)
      );
    } else {
      setSelectedParticipants([...selectedParticipants, userName]);
    }
  };

  // Helper to format working hours for display
  const formatWorkingHours = (start, end, days) => {
    if (!start || !end) return "Hours not set";
    const timeStr = (time) => {
      const timePart = time.split("+")[0].split("-")[0];
      const [h, m] = timePart.split(":");
      return `${h}:${m}`;
    };
    const daysStr = days && days.length > 0 ? days.join(", ") : "No days set";
    return `${daysStr} • ${timeStr(start)} - ${timeStr(end)}`;
  };

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
          participants: selectedParticipants,
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
          <Button variant="outline" onClick={() => setIsSettingsOpen(true)}>
            Set Working Hours
          </Button>
          <Button variant="outline">Meeting Alerts</Button>
          <Button onClick={handleLogout} variant="outline">
            Logout
          </Button>
        </div>
      </div>

      {/* Main Content - Calendar View */}
      <div className="flex-1 flex items-start justify-center overflow-hidden">
        <div className="w-full max-w-7xl h-full max-h-full">
          <Calendar username={username} />
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
              {/* Selected Participants Display in Chat */}
              {selectedParticipants.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-4 pb-4 border-b">
                  <span className="text-xs text-muted-foreground self-center mr-2">
                    Active Participants:
                  </span>
                  {selectedParticipants.map((p) => (
                    <Badge key={p} variant="secondary">
                      {p}
                    </Badge>
                  ))}
                </div>
              )}

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
          {/* Selected Participants Display */}
          {selectedParticipants.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3 max-w-4xl mx-auto px-1">
              <span className="text-xs text-muted-foreground self-center mr-2">
                Including:
              </span>
              {selectedParticipants.map((p) => (
                <Badge
                  key={p}
                  variant="secondary"
                  className="cursor-pointer hover:bg-destructive hover:text-destructive-foreground transition-colors"
                  onClick={() => toggleParticipant(p)}
                >
                  {p} ×
                </Badge>
              ))}
            </div>
          )}
          <div className="flex gap-2 max-w-4xl mx-auto">
            <Input
              placeholder="Type your request here (e.g., 'Schedule a meeting with John tomorrow at 2pm')..."
              className="flex-1"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <Button
              variant="outline"
              onClick={() => setIsParticipantsOpen(true)}
              className="relative"
            >
              Participants
              {selectedParticipants.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {selectedParticipants.length}
                </Badge>
              )}
            </Button>
            <Button
              onClick={handleSend}
              disabled={isProcessing || !inputValue.trim()}
            >
              Send
            </Button>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {isSettingsOpen && (
        <>
          <div
            className="fixed inset-0 bg-black/20 z-50 backdrop-blur-sm animate-in fade-in"
            onClick={() => setIsSettingsOpen(false)}
          />
          <Card className="fixed left-[50%] top-[50%] translate-x-[-50%] translate-y-[-50%] w-full max-w-md z-50 p-6 shadow-xl animate-in zoom-in-95 duration-200">
            <h2 className="text-xl font-bold mb-4">Working Hours</h2>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label className="text-sm font-medium">Working Days</Label>
                <div className="flex flex-wrap gap-2">
                  {daysOfWeek.map((day) => (
                    <Button
                      key={day}
                      variant={
                        workingDays.includes(day) ? "default" : "outline"
                      }
                      size="sm"
                      onClick={() => toggleDay(day)}
                      className="h-8 w-10 p-0"
                    >
                      {day}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="work-start">Start Time</Label>
                  <Input
                    id="work-start"
                    type="time"
                    value={workStart}
                    onChange={(e) => setWorkStart(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="work-end">End Time</Label>
                  <Input
                    id="work-end"
                    type="time"
                    value={workEnd}
                    onChange={(e) => setWorkEnd(e.target.value)}
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 mt-6">
                <Button
                  variant="ghost"
                  onClick={() => setIsSettingsOpen(false)}
                >
                  Cancel
                </Button>
                <Button onClick={handleSavePreferences} disabled={isSaving}>
                  {isSaving ? "Saving..." : "Save Changes"}
                </Button>
              </div>
            </div>
          </Card>
        </>
      )}

      {/* Success Modal */}
      <Dialog open={showSuccessModal} onOpenChange={setShowSuccessModal}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Success!</DialogTitle>
            <DialogDescription>
              Your working hours have been saved successfully.
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end">
            <Button onClick={() => setShowSuccessModal(false)}>
              Continue to Dashboard
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Participants Selection Modal */}
      <Dialog open={isParticipantsOpen} onOpenChange={setIsParticipantsOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Select Participants</DialogTitle>
            <DialogDescription>
              Choose who you want to include in the meeting.
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[60vh] overflow-y-auto py-4 space-y-4">
            {availableUsers.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                No other users found.
              </p>
            ) : (
              availableUsers.map((user) => (
                <div
                  key={user.name}
                  className="flex items-start space-x-3 p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                >
                  <Checkbox
                    id={`user-${user.name}`}
                    checked={selectedParticipants.includes(user.name)}
                    onCheckedChange={() => toggleParticipant(user.name)}
                    className="mt-1"
                  />
                  <div className="grid gap-1.5 leading-none">
                    <Label
                      htmlFor={`user-${user.name}`}
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                    >
                      {user.name}
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      {formatWorkingHours(
                        user.work_start,
                        user.work_end,
                        user.working_days
                      )}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setIsParticipantsOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={() => setIsParticipantsOpen(false)}>
              Done ({selectedParticipants.length})
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
