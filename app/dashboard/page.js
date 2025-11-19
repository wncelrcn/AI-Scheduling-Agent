"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Calendar } from "@/components/calendar/Calendar";
import { useEffect, useState } from "react";

export default function Dashboard() {
  const router = useRouter();
  const [username, setUsername] = useState("");

  useEffect(() => {
    // Check if user is logged in, redirect if not
    const storedUsername = sessionStorage.getItem("username");
    if (!storedUsername) {
      router.push("/");
    } else {
      setUsername(storedUsername);
    }
  }, [router]);

  const handleLogout = () => {
    sessionStorage.removeItem("username");
    router.push("/");
  };

  return (
    <div className="flex flex-col h-screen p-6">
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
      <div className="flex-1 flex items-start justify-center p-2 sm:p-4 overflow-hidden">
        <div className="w-full max-w-7xl h-full max-h-full">
          <Calendar />
        </div>
      </div>

      {/* Bottom Input Area */}
      <div className="mt-6 pt-4 border-t">
        <div className="flex gap-2 max-w-4xl mx-auto">
          <Input
            placeholder="Type your request here (e.g., 'Schedule a meeting with John tomorrow at 2pm')..."
            className="flex-1"
          />
          <Button>Send</Button>
        </div>
      </div>
    </div>
  );
}
