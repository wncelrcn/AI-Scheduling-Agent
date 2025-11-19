"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { useEffect } from "react";

export default function Dashboard() {
  const router = useRouter();

  useEffect(() => {
    // Check if user is logged in, redirect if not
    const storedUsername = sessionStorage.getItem("username");
    if (!storedUsername) {
      router.push("/");
    }
  }, [router]);

  const handleLogout = () => {
    sessionStorage.removeItem("username");
    router.push("/");
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">Calendar Dashboard</h1>
        <Button onClick={handleLogout} variant="outline">
          Logout
        </Button>
      </div>
      <p>
        Welcome,{" "}
        {typeof window !== "undefined" && sessionStorage.getItem("username")}!
      </p>
    </div>
  );
}
