"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { createClient } from "@/utils/supabase/client";

export default function Page() {
  const [username, setUsername] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const supabase = createClient();

  useEffect(() => {
    // Check if user is already logged in
    const storedUsername = sessionStorage.getItem("username");
    if (storedUsername) {
      router.push("/dashboard");
    }
  }, [router]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim()) return;

    setLoading(true);
    const name = username.trim();

    try {
      // Try to insert the user into the 'users' table
      const { error } = await supabase.from("users").insert({ name });

      if (error) {
        // Check for unique constraint violation (code 23505)
        // This means the user already exists, which is fine for this app
        if (error.code === "23505") {
          console.log("User already exists, logging in...");
        } else {
          // Handle other errors
          console.error("Error creating user:", error);
          alert("Error logging in: " + error.message);
          setLoading(false);
          return;
        }
      }

      // If successful or user exists, proceed to login
      sessionStorage.setItem("username", name);
      router.push("/dashboard");
    } catch (err) {
      console.error("Unexpected error:", err);
      alert("An unexpected error occurred.");
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-svh w-full items-center justify-center p-6 md:p-10">
      <div className="w-full max-w-sm">
        <Card>
          <CardHeader>
            <CardTitle>Welcome!</CardTitle>
            <CardDescription>
              Enter your Name to access the portal.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit}>
              <FieldGroup>
                <Field>
                  <FieldLabel htmlFor="username">Name</FieldLabel>
                  <Input
                    id="username"
                    type="text"
                    placeholder="Enter your Name"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                    autoFocus
                    disabled={loading}
                  />
                </Field>
                <Field>
                  <Button type="submit" className="w-full" disabled={loading}>
                    {loading ? "Logging In..." : "Log In"}
                  </Button>
                </Field>
              </FieldGroup>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
