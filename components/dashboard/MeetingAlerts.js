"use client";

import { useState, useEffect } from "react";
import { createClient } from "@/utils/supabase/client";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  CheckCircle2,
  XCircle,
  Clock,
  Calendar as CalendarIcon,
  Users,
  RefreshCcw,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

export function MeetingAlerts({ isOpen, onClose, username }) {
  const [activeTab, setActiveTab] = useState("incoming");
  const [incomingProposals, setIncomingProposals] = useState([]);
  const [myProposals, setMyProposals] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [rejectDialog, setRejectDialog] = useState({
    open: false,
    proposalId: null,
  });
  const [rejectReason, setRejectReason] = useState("");
  const [confirmationDialog, setConfirmationDialog] = useState({
    open: false,
    meetingData: null,
  });

  const supabase = createClient();

  useEffect(() => {
    if (isOpen && username) {
      fetchProposals();
    }
  }, [isOpen, username]);

  const fetchProposals = async () => {
    setIsLoading(true);
    try {
      // Fetch incoming proposals (where user is a participant)
      const { data: incomingData, error: incomingError } = await supabase
        .from("participant_responses")
        .select(
          `
          response,
          proposal_id,
          feedback,
          meeting_proposals (
            proposal_id,
            meeting_title,
            proposed_start,
            proposed_end,
            proposed_end,
            organizer_id,
            reasoning,
            priority_participants
          )
        `
        )
        .eq("participant_id", username)
        .neq("response", "rejected");

      if (incomingError) throw incomingError;

      const formattedIncoming = incomingData.map((item) => ({
        ...item.meeting_proposals,
        my_status: item.response,
        response_id: item.id,
        rejection_reason: item.feedback,
      }));
      setIncomingProposals(formattedIncoming);

      // Fetch my proposals (where user is organizer)
      const { data: myData, error: myError } = await supabase
        .from("meeting_proposals")
        .select(
          `
          *,
          participant_responses (
            participant_id,
            response,
            feedback
          )
        `
        )
        .eq("organizer_id", username)
        .order("created_at", { ascending: false });

      if (myError) throw myError;
      setMyProposals(myData);
    } catch (error) {
      console.error("Error fetching proposals:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAccept = async (proposalId) => {
    try {
      const { error } = await supabase
        .from("participant_responses")
        .update({ response: "accepted", feedback: null })
        .eq("proposal_id", proposalId)
        .eq("participant_id", username);

      if (error) throw error;
      fetchProposals();
    } catch (error) {
      console.error("Error accepting proposal:", error);
    }
  };

  const handleRejectClick = (proposalId) => {
    setRejectDialog({ open: true, proposalId });
    setRejectReason("");
  };

  const confirmReject = async () => {
    if (!rejectDialog.proposalId) return;

    try {
      // 1. Update participant_responses
      const { error: responseError } = await supabase
        .from("participant_responses")
        .update({
          response: "rejected",
          feedback: rejectReason,
        })
        .eq("proposal_id", rejectDialog.proposalId)
        .eq("participant_id", username);

      if (responseError) throw responseError;

      // 2. Update meeting_proposals (append feedback)
      // First fetch existing feedback to append
      const { data: proposalData, error: fetchError } = await supabase
        .from("meeting_proposals")
        .select("rejection_feedback, meeting_title")
        .eq("proposal_id", rejectDialog.proposalId)
        .single();

      if (!fetchError && proposalData) {
        const currentFeedback = proposalData.rejection_feedback || "";
        const newFeedbackEntry = `[${username}]: ${rejectReason}`;
        const updatedFeedback = currentFeedback
          ? `${currentFeedback}\n${newFeedbackEntry}`
          : newFeedbackEntry;

        await supabase
          .from("meeting_proposals")
          .update({ rejection_feedback: updatedFeedback })
          .eq("proposal_id", rejectDialog.proposalId);
      }

      // 3. Trigger Rescheduling Agent (REMOVED - now triggered manually by organizer)
      /*
      try {
        await fetch("http://localhost:8000/api/reschedule", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            proposal_id: rejectDialog.proposalId,
            feedback: rejectReason,
            username: username,
          }),
        });
      } catch (apiError) {
        console.error("Failed to trigger rescheduling:", apiError);
        // Don't block the UI update if the background agent trigger fails
      }
      */

      setRejectDialog({ open: false, proposalId: null });
      fetchProposals();
    } catch (error) {
      console.error("Error rejecting proposal:", error);
    }
  };

  const handleReschedule = async (proposal) => {
    setIsLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/reschedule", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          proposal_id: proposal.proposal_id,
          feedback:
            proposal.rejection_feedback || "Reschedule requested manually",
          username: username,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to trigger rescheduling");
      }

      // Refresh to show updated status
      fetchProposals();
    } catch (error) {
      console.error("Error rescheduling:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePushThrough = async (proposal) => {
    setIsLoading(true);
    try {
      const response = await fetch(
        "http://localhost:8000/api/finalize_meeting",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            proposal_id: proposal.proposal_id,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to finalize meeting");
      }

      const data = await response.json();

      // Show confirmation modal instead of alert
      setConfirmationDialog({
        open: true,
        meetingData: {
          title: proposal.meeting_title,
          attendees: data.attendees || [],
          start: proposal.proposed_start,
          end: proposal.proposed_end,
        },
      });

      // Refresh to show updated status
      fetchProposals();
    } catch (error) {
      console.error("Error finalizing meeting:", error);
      // Show error in confirmation dialog
      setConfirmationDialog({
        open: true,
        meetingData: {
          error: true,
          message: "Failed to finalize meeting. Please try again.",
        },
      });
    } finally {
      setIsLoading(false);
    }
  };

  const calculateMajority = (responses, priorityParticipants = []) => {
    if (!responses || responses.length === 0) return false;

    // Check if any priority participant has rejected
    const priorityRejection = responses.some(
      (r) =>
        r.response === "rejected" &&
        priorityParticipants &&
        priorityParticipants.includes(r.participant_id)
    );

    if (priorityRejection) return false;

    const totalParticipants = responses.length;
    const acceptedParticipants = responses.filter(
      (r) => r.response === "accepted"
    ).length;

    // Total people involved = Participants + Organizer (1)
    const totalPeople = totalParticipants + 1;

    // Total accepted = Accepted Participants + Organizer (1)
    const totalAccepted = acceptedParticipants + 1;

    // Majority is > 50%
    return totalAccepted > totalPeople / 2;
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
    }).format(date);
  };

  const formatDateRange = (startStr, endStr) => {
    const start = new Date(startStr);
    const end = new Date(endStr);
    const startFormatted = formatDate(startStr);

    if (start.toDateString() === end.toDateString()) {
      const endTime = new Intl.DateTimeFormat("en-US", {
        hour: "numeric",
        minute: "numeric",
      }).format(end);
      return `${startFormatted} - ${endTime}`;
    }

    return `${startFormatted} - ${formatDate(endStr)}`;
  };

  const formatStatus = (status) => {
    if (!status) return "Unknown";
    return status.charAt(0).toUpperCase() + status.slice(1);
  };

  const getStatusBadgeVariant = (status) => {
    switch (status) {
      case "accepted":
        return "default"; // or a specific green variant if available, defaulting to default (primary)
      case "rejected":
        return "destructive";
      case "pending":
        return "secondary";
      default:
        return "outline";
    }
  };

  const getStatusColorClass = (status) => {
    switch (status) {
      case "accepted":
        return "bg-green-500 hover:bg-green-600";
      case "pending":
        return "bg-yellow-500 hover:bg-yellow-600 text-white";
      default:
        return "";
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[800px] h-[80vh] flex flex-col p-0 gap-0 overflow-hidden">
        <div className="p-6 pb-4 border-b">
          <DialogHeader>
            <DialogTitle className="text-2xl flex items-center gap-2">
              <Clock className="w-6 h-6 text-primary" />
              Meeting Alerts
            </DialogTitle>
            <div className="flex items-center justify-between">
              <DialogDescription>
                Manage your meeting proposals and invitations.
              </DialogDescription>
              <Button
                variant="ghost"
                size="icon"
                onClick={fetchProposals}
                disabled={isLoading}
                aria-label="Refresh alerts"
              >
                <RefreshCcw
                  className={cn("h-4 w-4", isLoading && "animate-spin")}
                />
              </Button>
            </div>
          </DialogHeader>
        </div>

        {/* Shadcn-like Tabs Implementation */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="px-6 pt-4 pb-2">
            <div className="inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground w-full">
              <button
                onClick={() => setActiveTab("incoming")}
                className={cn(
                  "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 w-1/2",
                  activeTab === "incoming"
                    ? "bg-background text-foreground shadow-sm"
                    : "hover:bg-background/50 hover:text-foreground"
                )}
              >
                Incoming Requests
              </button>
              <button
                onClick={() => setActiveTab("my-proposals")}
                className={cn(
                  "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 w-1/2",
                  activeTab === "my-proposals"
                    ? "bg-background text-foreground shadow-sm"
                    : "hover:bg-background/50 hover:text-foreground"
                )}
              >
                My Proposals
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto bg-muted/30 p-6">
            {activeTab === "incoming" && (
              <div className="space-y-5">
                {incomingProposals.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 px-4">
                    <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                      <CalendarIcon className="w-10 h-10 text-primary/40" />
                    </div>
                    <h3 className="text-lg font-semibold text-foreground mb-2">
                      All caught up!
                    </h3>
                    <p className="text-sm text-muted-foreground text-center max-w-sm">
                      No incoming meeting requests at the moment. You'll see new
                      invitations here.
                    </p>
                  </div>
                ) : (
                  incomingProposals.map((proposal) => (
                    <Card
                      key={proposal.proposal_id}
                      className={cn(
                        "group relative overflow-hidden transition-all duration-200 hover:shadow-lg border-l-4",
                        proposal.my_status === "pending"
                          ? "border-l-blue-500 bg-gradient-to-r from-blue-50/50 via-background to-background dark:from-blue-950/20 dark:via-background dark:to-background"
                          : proposal.my_status === "accepted"
                          ? "border-l-green-500 bg-gradient-to-r from-green-50/30 via-background to-background dark:from-green-950/20 dark:via-background dark:to-background"
                          : "border-l-muted"
                      )}
                    >
                      <CardHeader className="pb-4 space-y-3">
                        <div className="flex justify-between items-start gap-4">
                          <div className="flex-1 min-w-0">
                            <CardTitle className="text-xl font-bold tracking-tight mb-1.5 flex items-center gap-2">
                              {proposal.meeting_title || "Untitled Meeting"}
                              {proposal.my_status === "pending" && (
                                <span className="relative flex h-2 w-2">
                                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                                  <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                                </span>
                              )}
                            </CardTitle>
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <Users className="w-3.5 h-3.5" />
                              <span>
                                Organized by{" "}
                                <span className="font-medium text-foreground">
                                  {proposal.organizer_id}
                                </span>
                              </span>
                            </div>
                          </div>
                          <Badge
                            variant={getStatusBadgeVariant(proposal.my_status)}
                            className={cn(
                              "capitalize font-semibold px-3 py-1.5 text-xs shrink-0",
                              proposal.my_status === "accepted" &&
                                "bg-green-500/90 hover:bg-green-500 text-white border-green-600",
                              proposal.my_status === "pending" &&
                                "bg-blue-500/90 hover:bg-blue-500 text-white border-blue-600 animate-pulse"
                            )}
                          >
                            {proposal.my_status === "accepted" && (
                              <CheckCircle2 className="w-3 h-3 mr-1 inline" />
                            )}
                            {proposal.my_status === "pending" && (
                              <Clock className="w-3 h-3 mr-1 inline" />
                            )}
                            {formatStatus(proposal.my_status)}
                          </Badge>
                        </div>
                      </CardHeader>

                      <CardContent className="pb-4">
                        <div className="rounded-lg bg-muted/50 p-4 border border-border/50">
                          <div className="flex items-start gap-3">
                            <div className="rounded-md bg-primary/10 p-2 shrink-0">
                              <CalendarIcon className="w-5 h-5 text-primary" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
                                Scheduled Time
                              </p>
                              <p className="text-sm font-semibold text-foreground leading-relaxed">
                                {formatDate(proposal.proposed_start)}
                              </p>
                              <p className="text-xs text-muted-foreground mt-0.5">
                                Until {formatDate(proposal.proposed_end)}
                              </p>
                            </div>
                          </div>
                        </div>
                      </CardContent>

                      <CardFooter
                        className={cn(
                          "border-t p-4 transition-colors",
                          proposal.my_status === "pending"
                            ? "bg-muted/30"
                            : "bg-muted/20"
                        )}
                      >
                        {proposal.my_status === "pending" ? (
                          <div className="flex gap-3 w-full">
                            <Button
                              variant="outline"
                              size="default"
                              onClick={() =>
                                handleRejectClick(proposal.proposal_id)
                              }
                              className="flex-1 text-rose-600 hover:text-rose-700 hover:bg-rose-50 border-rose-200 hover:border-rose-300 dark:hover:bg-rose-950/20 dark:border-rose-900 transition-all duration-200 font-semibold"
                            >
                              <XCircle className="w-4 h-4 mr-2" />
                              Decline
                            </Button>
                            <Button
                              size="default"
                              onClick={() => handleAccept(proposal.proposal_id)}
                              className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white shadow-md hover:shadow-lg transition-all duration-200 font-semibold"
                            >
                              <CheckCircle2 className="w-4 h-4 mr-2" />
                              Accept Meeting
                            </Button>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 text-sm w-full">
                            {proposal.my_status === "accepted" && (
                              <div className="flex items-center gap-2 text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-950/30 px-3 py-2 rounded-md w-full">
                                <CheckCircle2 className="w-4 h-4" />
                                <span className="font-medium">
                                  You accepted this meeting
                                </span>
                              </div>
                            )}
                          </div>
                        )}
                      </CardFooter>
                    </Card>
                  ))
                )}
              </div>
            )}

            {activeTab === "my-proposals" && (
              <div className="space-y-4">
                {myProposals.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <p>You haven't proposed any meetings yet.</p>
                  </div>
                ) : (
                  myProposals.map((proposal) => (
                    <Card key={proposal.proposal_id}>
                      <CardHeader className="pb-3">
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle className="text-lg">
                              {proposal.meeting_title || "Untitled Meeting"}
                            </CardTitle>
                            <CardDescription>
                              {formatDateRange(
                                proposal.proposed_start,
                                proposal.proposed_end
                              )}
                            </CardDescription>
                          </div>
                          <Badge
                            variant={
                              proposal.status === "confirmed"
                                ? "default"
                                : "secondary"
                            }
                          >
                            {formatStatus(proposal.status)}
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-4">
                          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                            <Users className="w-4 h-4" />
                            <span>Participants Status</span>
                          </div>
                          <div className="grid gap-3">
                            {proposal.participant_responses?.map((resp) => (
                              <div
                                key={resp.participant_id}
                                className="flex flex-col gap-2 bg-muted/30 p-3 rounded-md border"
                              >
                                <div className="flex items-center justify-between text-sm">
                                  <span className="font-medium">
                                    {resp.participant_id}
                                  </span>
                                  <div className="flex items-center gap-2">
                                    {resp.response === "accepted" && (
                                      <CheckCircle2 className="w-4 h-4 text-green-500" />
                                    )}
                                    {resp.response === "rejected" && (
                                      <XCircle className="w-4 h-4 text-destructive" />
                                    )}
                                    {resp.response === "pending" && (
                                      <Clock className="w-4 h-4 text-yellow-500" />
                                    )}
                                    <Badge
                                      variant="outline"
                                      className="text-xs font-normal capitalize"
                                    >
                                      {formatStatus(resp.response)}
                                    </Badge>
                                  </div>
                                </div>
                                {resp.response === "rejected" && (
                                  <div className="text-sm bg-destructive/10 text-destructive p-2 rounded border border-destructive/20">
                                    <span className="font-semibold block text-xs mb-1">
                                      Reason for Rejection:
                                    </span>
                                    {resp.feedback || "No reason provided"}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>

                          {/* Re-generate Schedule Button if any rejection */}
                          {proposal.participant_responses?.some(
                            (r) => r.response === "rejected"
                          ) &&
                            proposal.status === "pending" && (
                              <div className="pt-2 flex flex-wrap gap-2">
                                <Button
                                  className="w-full sm:w-auto gap-2"
                                  variant="secondary"
                                  onClick={() => handleReschedule(proposal)}
                                  disabled={isLoading}
                                >
                                  <RefreshCcw
                                    className={cn(
                                      "w-4 h-4",
                                      isLoading && "animate-spin"
                                    )}
                                  />
                                  Re-generate Schedule
                                </Button>

                                {calculateMajority(
                                  proposal.participant_responses,
                                  proposal.priority_participants
                                ) ? (
                                  <Button
                                    className="w-full sm:w-auto gap-2 bg-orange-500 hover:bg-orange-600 text-white"
                                    variant="default"
                                    onClick={() => handlePushThrough(proposal)}
                                    disabled={isLoading}
                                  >
                                    <Users className="w-4 h-4" />
                                    Push Through (Majority Accepted)
                                  </Button>
                                ) : (
                                  proposal.participant_responses?.some(
                                    (r) =>
                                      r.response === "rejected" &&
                                      proposal.priority_participants?.includes(
                                        r.participant_id
                                      )
                                  ) && (
                                    <div className="flex items-center gap-2 text-xs text-destructive bg-destructive/10 px-3 py-2 rounded border border-destructive/20">
                                      <AlertCircle className="w-4 h-4" />
                                      <span>
                                        Cannot push through: Priority
                                        participant rejected.
                                      </span>
                                    </div>
                                  )
                                )}
                              </div>
                            )}
                        </div>
                      </CardContent>
                    </Card>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      </DialogContent>

      {/* Reject/Negotiate Dialog */}
      <Dialog
        open={rejectDialog.open}
        onOpenChange={(open) =>
          !open && setRejectDialog({ open: false, proposalId: null })
        }
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Decline Meeting</DialogTitle>
            <DialogDescription>
              Please let the organizer know why you can't make it, or suggest an
              alternative time.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label htmlFor="reason" className="mb-2 block">
              Reason / Constraints
            </Label>
            <textarea
              id="reason"
              placeholder="I have a conflict at this time. Could we do 2 PM instead?"
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              className="flex min-h-[100px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setRejectDialog({ open: false, proposalId: null })}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmReject}>
              Send Response
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Meeting Confirmation Dialog */}
      <Dialog
        open={confirmationDialog.open}
        onOpenChange={(open) =>
          !open && setConfirmationDialog({ open: false, meetingData: null })
        }
      >
        <DialogContent className="sm:max-w-md">
          {confirmationDialog.meetingData?.error ? (
            <>
              <DialogHeader>
                <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-full bg-destructive/10">
                  <XCircle className="w-8 h-8 text-destructive" />
                </div>
                <DialogTitle className="text-center text-xl">
                  Failed to Finalize Meeting
                </DialogTitle>
                <DialogDescription className="text-center">
                  {confirmationDialog.meetingData.message}
                </DialogDescription>
              </DialogHeader>
              <DialogFooter className="sm:justify-center">
                <Button
                  onClick={() =>
                    setConfirmationDialog({ open: false, meetingData: null })
                  }
                >
                  Close
                </Button>
              </DialogFooter>
            </>
          ) : (
            <>
              <DialogHeader>
                <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-full bg-green-100 dark:bg-green-950">
                  <CheckCircle2 className="w-8 h-8 text-green-600 dark:text-green-400" />
                </div>
                <DialogTitle className="text-center text-xl">
                  Meeting Confirmed! 🎉
                </DialogTitle>
                <DialogDescription className="text-center">
                  The meeting has been successfully scheduled.
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4 py-4">
                {/* Meeting Title */}
                <div className="bg-muted/50 rounded-lg p-4 border">
                  <h3 className="font-semibold text-sm text-muted-foreground mb-1">
                    Meeting
                  </h3>
                  <p className="text-base font-medium">
                    {confirmationDialog.meetingData?.title ||
                      "Untitled Meeting"}
                  </p>
                </div>

                {/* Time */}
                <div className="bg-muted/50 rounded-lg p-4 border">
                  <div className="flex items-center gap-2 mb-2">
                    <CalendarIcon className="w-4 h-4 text-primary" />
                    <h3 className="font-semibold text-sm text-muted-foreground">
                      Scheduled Time
                    </h3>
                  </div>
                  <p className="text-sm">
                    {confirmationDialog.meetingData?.start &&
                      formatDateRange(
                        confirmationDialog.meetingData.start,
                        confirmationDialog.meetingData.end
                      )}
                  </p>
                </div>

                {/* Attendees */}
                <div className="bg-muted/50 rounded-lg p-4 border">
                  <div className="flex items-center gap-2 mb-2">
                    <Users className="w-4 h-4 text-primary" />
                    <h3 className="font-semibold text-sm text-muted-foreground">
                      Confirmed Attendees
                    </h3>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant="secondary"
                      className="text-base font-semibold"
                    >
                      {confirmationDialog.meetingData?.attendees?.length || 0}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      participants confirmed
                    </span>
                  </div>
                </div>
              </div>

              <DialogFooter className="sm:justify-center">
                <Button
                  className="w-full sm:w-auto"
                  onClick={() =>
                    setConfirmationDialog({ open: false, meetingData: null })
                  }
                >
                  Got it, thanks!
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>
    </Dialog>
  );
}
