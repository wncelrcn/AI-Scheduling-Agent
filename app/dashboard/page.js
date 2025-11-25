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
import {
  Send,
  Users,
  X,
  Bot,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Star,
} from "lucide-react";

import { MeetingAlerts } from "@/components/dashboard/MeetingAlerts";

export default function Dashboard() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState([]);
  const [isChatOpen, setIsChatOpen] = useState(false); // Controls full expansion (history)
  const [isSheetPeeking, setIsSheetPeeking] = useState(false); // Controls input visibility
  const [isProcessing, setIsProcessing] = useState(false);
  const [agentState, setAgentState] = useState(null); // Store agent state for continuity
  const scrollRef = useRef(null);

  // Settings State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isPreferencesIncomplete, setIsPreferencesIncomplete] = useState(false);
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
  const [selectedPriorityParticipants, setSelectedPriorityParticipants] = useState([]);
  const [isParticipantsOpen, setIsParticipantsOpen] = useState(false);

  // Meeting Alerts State
  const [isMeetingAlertsOpen, setIsMeetingAlertsOpen] = useState(false);

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
        // Check if preferences are incomplete
        if (
          !data.work_start ||
          !data.work_end ||
          !data.working_days ||
          data.working_days.length === 0
        ) {
          setIsPreferencesIncomplete(true);
        } else {
          setIsPreferencesIncomplete(false);
        }

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
      setIsPreferencesIncomplete(false);
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
      // Also remove from priority if removed from selection
      if (selectedPriorityParticipants.includes(userName)) {
        setSelectedPriorityParticipants(
          selectedPriorityParticipants.filter((p) => p !== userName)
        );
      }
    } else {
      setSelectedParticipants([...selectedParticipants, userName]);
    }
  };

  const togglePriority = (userName, e) => {
    e.stopPropagation(); // Prevent toggling participant selection
    if (selectedPriorityParticipants.includes(userName)) {
      setSelectedPriorityParticipants(
        selectedPriorityParticipants.filter((p) => p !== userName)
      );
    } else {
      setSelectedPriorityParticipants([...selectedPriorityParticipants, userName]);
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

    if (selectedParticipants.length === 0) {
      setIsParticipantsOpen(true);
      return;
    }

    const userMessage = {
      role: "user",
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    // Optimistic update
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsChatOpen(true); // Auto-expand chat on send
    setIsSheetPeeking(true); // Ensure sheet is visible
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
          history: history,
          participants: selectedParticipants,
          priority_participants: selectedPriorityParticipants,
          previous_state: agentState, // Pass previous agent state for continuity
        }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();

      // Store agent state for next turn
      if (data.agent_state) {
        setAgentState(data.agent_state);
      }

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

  // Handle toggle behavior
  const toggleSheet = () => {
    if (isChatOpen) {
      // If fully open, minimize completely
      setIsChatOpen(false);
      setIsSheetPeeking(false);
    } else if (isSheetPeeking) {
      // If peeking (input visible), expand to full
      setIsChatOpen(true);
    } else {
      // If minimized, peek (show input)
      setIsSheetPeeking(true);
    }
  };

  // Explicitly close
  const closeSheet = () => {
    setIsChatOpen(false);
    setIsSheetPeeking(false);
  };

  // Determine transform class based on state
  const getSheetTransform = () => {
    if (isChatOpen) return "translate-y-0"; // Full height
    if (isSheetPeeking) return "translate-y-0"; // Auto height (content visible)
    return "translate-y-[calc(100%-24px)]"; // Only handle visible (approx 24px)
  };

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden relative">
      {/* Top Section with Calendar */}
      <div className="flex-1 flex flex-col p-6 pb-12 overflow-hidden relative z-0">
        {/* Header */}
        <div className="flex justify-between items-center mb-6 shrink-0">
          <div>
            <h1 className="text-2xl font-bold">Calendar Dashboard</h1>
            <p className="text-sm text-muted-foreground">
              Welcome back, {username}!
            </p>
          </div>
          <div className="flex gap-4">
            <Button
              variant={isPreferencesIncomplete ? "default" : "outline"}
              className={
                isPreferencesIncomplete
                  ? "bg-yellow-500 hover:bg-yellow-600 text-white border-yellow-600 animate-pulse"
                  : ""
              }
              onClick={() => setIsSettingsOpen(true)}
            >
              Set Working Hours
              {isPreferencesIncomplete && (
                <AlertCircle className="ml-2 h-4 w-4" />
              )}
            </Button>
            <Button
              variant="outline"
              onClick={() => setIsMeetingAlertsOpen(true)}
            >
              Meeting Alerts
            </Button>
            <Button onClick={handleLogout} variant="outline">
              Logout
            </Button>
          </div>
        </div>

        {/* Preferences Alert Banner */}
        {isPreferencesIncomplete && (
          <div className="mb-6 bg-yellow-500/10 border border-yellow-500/50 text-yellow-700 dark:text-yellow-400 px-4 py-3 rounded-lg flex items-center justify-between animate-in slide-in-from-top-2">
            <div className="flex items-center gap-3">
              <AlertCircle className="h-5 w-5 flex-shrink-0" />
              <div>
                <p className="font-medium">Complete your profile</p>
                <p className="text-sm opacity-90">
                  Set your working hours and days to help the AI schedule
                  meetings effectively.
                </p>
              </div>
            </div>
            <Button
              size="sm"
              variant="secondary"
              className="ml-4 bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-700 dark:text-yellow-300 border-yellow-500/20 whitespace-nowrap"
              onClick={() => setIsSettingsOpen(true)}
            >
              Set Now
            </Button>
          </div>
        )}

        {/* Main Content - Calendar View */}
        <div className="flex-1 flex items-start justify-center overflow-hidden">
          <div className="w-full max-w-7xl h-full max-h-full">
            <Calendar username={username} />
          </div>
        </div>
      </div>

      {/* Unified Bottom Sheet Assistant */}
      <div
        className={`fixed bottom-0 left-0 right-0 bg-background border-t shadow-[0_-5px_30px_rgba(0,0,0,0.15)] transition-all duration-500 cubic-bezier(0.32, 0.72, 0, 1) z-50 flex flex-col ${
          isChatOpen ? "h-[75vh]" : "h-auto"
        } ${getSheetTransform()}`}
      >
        {/* Toggle Handle Area */}
        <div
          className="w-full flex flex-col items-center justify-center py-2 cursor-pointer hover:bg-muted/50 transition-colors active:bg-muted shrink-0 border-b border-transparent hover:border-border group h-8 relative"
          onClick={toggleSheet}
          title={isSheetPeeking ? "Expand or Collapse" : "Open AI Assistant"}
        >
          {/* Indicator Line */}
          <div className="w-12 h-1.5 bg-muted-foreground/20 rounded-full transition-colors group-hover:bg-primary/40 mb-1" />

          {/* Minimized Text (Always Visible) */}
          {!isSheetPeeking && !isChatOpen && (
            <span className="text-[10px] font-medium text-muted-foreground/70 uppercase tracking-widest animate-pulse">
              Click here to Schedule Meetings
            </span>
          )}
        </div>

        {/* Expanded Content: Chat History */}
        <div
          className={`flex-1 overflow-hidden flex flex-col bg-muted/5 transition-all duration-300 ${
            isChatOpen ? "opacity-100" : "opacity-0 hidden"
          }`}
        >
          <div className="px-6 py-2 border-b flex justify-between items-center bg-background/50 backdrop-blur supports-[backdrop-filter]:bg-background/50">
            <div className="flex items-center gap-2">
              <Bot className="w-4 h-4 text-primary" />
              <span className="font-semibold text-sm">
                AI Scheduling Assistant
              </span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => setIsChatOpen(false)}
            >
              <ChevronDown className="w-4 h-4" />
            </Button>
          </div>

          <CardContent
            className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth"
            ref={scrollRef}
          >
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-muted-foreground opacity-50">
                <p>Start a conversation to schedule meetings.</p>
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
        </div>

        {/* Always Visible Bottom Section: Input & Context */}
        <div
          className={`p-4 pt-2 bg-background relative z-10 shrink-0 flex flex-col gap-3 pb-6 sm:pb-8 transition-opacity duration-300 ${
            !isSheetPeeking && !isChatOpen ? "opacity-0" : "opacity-100"
          }`}
        >
          {/* Collapsed State Title (Only show when collapsed) */}
          {!isChatOpen && (
            <div className="flex items-center gap-2 px-1 animate-in fade-in duration-300">
              <Bot className="w-5 h-5 text-primary" />
              <h2 className="text-sm font-semibold">AI Scheduling Assistant</h2>
              <span className="text-xs text-muted-foreground ml-auto">
                Swipe up to expand
              </span>
            </div>
          )}

          {/* Selected Participants Chips */}
          <div className="min-h-[28px] flex items-center">
            {selectedParticipants.length > 0 ? (
              <div className="flex flex-wrap gap-2 animate-in fade-in slide-in-from-bottom-1 w-full">
                {!isChatOpen && (
                  <span className="text-xs text-muted-foreground mr-1">
                    With:
                  </span>
                )}
                {selectedParticipants.map((p) => (
                  <Badge
                    key={p}
                    variant="secondary"
                    className="cursor-pointer hover:bg-destructive hover:text-destructive-foreground transition-colors px-2 py-0.5 text-xs flex items-center gap-1"
                    onClick={() => toggleParticipant(p)}
                  >
                    {p} 
                    {selectedPriorityParticipants.includes(p) && (
                      <Star className="w-3 h-3 fill-yellow-500 text-yellow-500 ml-0.5" />
                    )}
                    <X className="w-3 h-3 opacity-50 ml-1" />
                  </Badge>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground flex items-center gap-2">
                <AlertCircle className="w-3 h-3" />
                Select participants to enable chat
              </p>
            )}
          </div>

          {/* Input Row */}
          <div className="flex gap-3 items-stretch">
            <Button
              variant={
                selectedParticipants.length === 0 ? "default" : "outline"
              }
              size="default"
              onClick={() => setIsParticipantsOpen(true)}
              className={`shadow-sm transition-all ${
                selectedParticipants.length === 0
                  ? "ring-2 ring-primary/20"
                  : ""
              }`}
            >
              <Users className="w-4 h-4 mr-2" />
              <span className="hidden sm:inline">Participants</span>
              <span className="sm:hidden">Add</span>
              {selectedParticipants.length > 0 && (
                <Badge
                  variant="secondary"
                  className="ml-2 bg-primary/10 text-primary border-0 h-5 px-1.5"
                >
                  {selectedParticipants.length}
                </Badge>
              )}
            </Button>

            <div className="flex-1 flex gap-2">
              <Input
                placeholder={
                  selectedParticipants.length === 0
                    ? "Select participants..."
                    : "Type your request..."
                }
                className="flex-1 shadow-sm"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={selectedParticipants.length === 0}
                onFocus={() => {
                  setIsChatOpen(true);
                  setIsSheetPeeking(true);
                }}
              />
              <Button
                onClick={handleSend}
                disabled={
                  isProcessing ||
                  !inputValue.trim() ||
                  selectedParticipants.length === 0
                }
                className="shadow-sm px-4"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>

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
                  className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-start space-x-3 flex-1">
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
                  
                  {/* Priority Toggle */}
                  {selectedParticipants.includes(user.name) && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={(e) => togglePriority(user.name, e)}
                      title={selectedPriorityParticipants.includes(user.name) ? "Unmark as Priority" : "Mark as Priority"}
                    >
                      <Star 
                        className={`w-4 h-4 ${selectedPriorityParticipants.includes(user.name) ? "fill-yellow-500 text-yellow-500" : "text-muted-foreground"}`} 
                      />
                    </Button>
                  )}
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

      {/* Meeting Alerts Modal */}
      <MeetingAlerts
        isOpen={isMeetingAlertsOpen}
        onClose={() => setIsMeetingAlertsOpen(false)}
        username={username}
      />
    </div>
  );
}
