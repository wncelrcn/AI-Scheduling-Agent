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
  const [rescheduleConfirmDialog, setRescheduleConfirmDialog] = useState({
    open: false,
    proposal: null,
    slotIndex: null,
  });
  const [rescheduleRejectDialog, setRescheduleRejectDialog] = useState({
    open: false,
    proposalId: null,
  });
  const [organizerFeedback, setOrganizerFeedback] = useState("");

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
        .eq("participant_id", username);

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

  const handleConfirmReschedule = async (proposal, slotIndex) => {
    setIsLoading(true);
    try {
      const response = await fetch(
        "http://localhost:8000/api/confirm_reschedule",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            proposal_id: proposal.proposal_id,
            selected_slot_index: slotIndex,
            username: username,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to confirm reschedule");
      }

      const data = await response.json();

      // Show success message
      setConfirmationDialog({
        open: true,
        meetingData: {
          title: proposal.meeting_title,
          attendees: proposal.participant_ids || [],
          start: data.confirmed_slot.start,
          end: data.confirmed_slot.end,
          isReschedule: true,
        },
      });

      // Close the reschedule confirm dialog
      setRescheduleConfirmDialog({
        open: false,
        proposal: null,
        slotIndex: null,
      });

      // Refresh proposals
      fetchProposals();
    } catch (error) {
      console.error("Error confirming reschedule:", error);
      alert("Failed to confirm reschedule. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRejectRescheduleClick = (proposalId) => {
    setRescheduleRejectDialog({ open: true, proposalId });
    setOrganizerFeedback("");
  };

  const confirmRejectReschedule = async () => {
    if (!rescheduleRejectDialog.proposalId) return;

    setIsLoading(true);
    try {
      const response = await fetch(
        "http://localhost:8000/api/reject_reschedule",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            proposal_id: rescheduleRejectDialog.proposalId,
            organizer_feedback: organizerFeedback,
            username: username,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to reject reschedule");
      }

      // Close dialog
      setRescheduleRejectDialog({ open: false, proposalId: null });

      // Refresh to show new suggestions
      fetchProposals();
    } catch (error) {
      console.error("Error rejecting reschedule:", error);
      alert("Failed to process feedback. Please try again.");
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
              <Clock className="w-6 h-6 text-foreground" />
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
                    <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mb-4">
                      <CalendarIcon className="w-10 h-10 text-muted-foreground" />
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
                        "group relative overflow-hidden transition-all duration-200 hover:shadow-md border-l-4",
                        proposal.my_status === "pending"
                          ? "border-l-foreground/60 bg-muted/30"
                          : proposal.my_status === "accepted"
                          ? "border-l-foreground/40 bg-muted/20"
                          : proposal.my_status === "rejected"
                          ? "border-l-foreground/20 bg-muted/10 opacity-70"
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
                                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-foreground/40 opacity-75"></span>
                                  <span className="relative inline-flex rounded-full h-2 w-2 bg-foreground"></span>
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
                            variant={
                              proposal.my_status === "pending"
                                ? "default"
                                : "secondary"
                            }
                            className={cn(
                              "capitalize font-semibold px-3 py-1.5 text-xs shrink-0",
                              proposal.my_status === "pending" &&
                                "animate-pulse",
                              proposal.my_status === "rejected" && "opacity-60"
                            )}
                          >
                            {proposal.my_status === "accepted" && (
                              <CheckCircle2 className="w-3 h-3 mr-1 inline" />
                            )}
                            {proposal.my_status === "pending" && (
                              <Clock className="w-3 h-3 mr-1 inline" />
                            )}
                            {proposal.my_status === "rejected" && (
                              <XCircle className="w-3 h-3 mr-1 inline" />
                            )}
                            {formatStatus(proposal.my_status)}
                          </Badge>
                        </div>
                      </CardHeader>

                      <CardContent className="pb-4">
                        <div className="rounded-lg bg-muted/50 p-4 border border-border/50">
                          <div className="flex items-start gap-3">
                            <div className="rounded-md bg-foreground/10 p-2 shrink-0">
                              <CalendarIcon className="w-5 h-5 text-foreground" />
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
                              className="flex-1 font-semibold"
                            >
                              <XCircle className="w-4 h-4 mr-2" />
                              Decline
                            </Button>
                            <Button
                              size="default"
                              onClick={() => handleAccept(proposal.proposal_id)}
                              className="flex-1 font-semibold"
                            >
                              <CheckCircle2 className="w-4 h-4 mr-2" />
                              Accept Meeting
                            </Button>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 text-sm w-full">
                            {proposal.my_status === "accepted" && (
                              <div className="flex items-center gap-2 bg-muted px-3 py-2 rounded-md w-full">
                                <CheckCircle2 className="w-4 h-4" />
                                <span className="font-medium">
                                  You accepted this meeting
                                </span>
                              </div>
                            )}
                            {proposal.my_status === "rejected" && (
                              <div className="flex items-center gap-2 bg-muted px-3 py-2 rounded-md w-full opacity-70">
                                <XCircle className="w-4 h-4" />
                                <span className="font-medium">
                                  You declined this meeting
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
              <div className="space-y-5">
                {myProposals.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-16 px-4">
                    <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mb-4">
                      <Users className="w-10 h-10 text-muted-foreground" />
                    </div>
                    <h3 className="text-lg font-semibold text-foreground mb-2">
                      No Proposals Yet
                    </h3>
                    <p className="text-sm text-muted-foreground text-center max-w-sm">
                      You haven't organized any meetings yet. Start a
                      conversation in the scheduling assistant to create your
                      first proposal.
                    </p>
                  </div>
                ) : (
                  myProposals.map((proposal) => (
                    <Card
                      key={proposal.proposal_id}
                      className="overflow-hidden border-2 hover:shadow-lg transition-all duration-200"
                    >
                      <CardHeader className="pb-4 bg-gradient-to-r from-muted/30 to-background">
                        <div className="flex justify-between items-start gap-4">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-2">
                              <div className="rounded-lg bg-foreground/10 p-2">
                                <CalendarIcon className="w-4 h-4 text-foreground" />
                              </div>
                              <CardTitle className="text-xl font-bold">
                              {proposal.meeting_title || "Untitled Meeting"}
                            </CardTitle>
                            </div>
                            <CardDescription className="flex items-center gap-2 text-base">
                              <Clock className="w-4 h-4" />
                              {formatDateRange(
                                proposal.proposed_start,
                                proposal.proposed_end
                              )}
                            </CardDescription>
                          </div>
                          <div className="shrink-0">
                            {proposal.status ===
                            "awaiting_organizer_approval" ? (
                              <Badge className="animate-pulse px-3 py-1.5">
                                <Clock className="w-3.5 h-3.5 mr-1.5 inline" />
                                Needs Review
                              </Badge>
                            ) : proposal.status === "finalized" ? (
                          <Badge
                                variant="secondary"
                                className="px-3 py-1.5"
                              >
                                <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 inline" />
                                Confirmed
                              </Badge>
                            ) : (
                              <Badge
                                variant="secondary"
                                className="px-3 py-1.5 font-semibold"
                          >
                            {formatStatus(proposal.status)}
                          </Badge>
                            )}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        {/* Reschedule Suggestions Section */}
                        {proposal.status === "awaiting_organizer_approval" &&
                          proposal.suggested_slots && (
                            <div className="space-y-6 mb-6 p-6 bg-muted/50 dark:bg-muted/30 rounded-lg border-2 border-border">
                              {/* Header Banner */}
                              <div className="flex items-start gap-4 pb-5 border-b border-border">
                                <div className="rounded-lg bg-foreground/10 p-3 shrink-0">
                                  <RefreshCcw className="w-6 h-6 text-foreground" />
                                </div>
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-2">
                                    <h3 className="text-xl font-bold text-foreground">
                                      Alternative Time Slots Ready
                                    </h3>
                                    <Badge
                                      variant="secondary"
                                      className="font-semibold"
                                    >
                                      {(() => {
                                        let suggestedSlotsData =
                                          proposal.suggested_slots;
                                        if (
                                          typeof suggestedSlotsData === "string"
                                        ) {
                                          try {
                                            suggestedSlotsData =
                                              JSON.parse(suggestedSlotsData);
                                          } catch (e) {
                                            return "0";
                                          }
                                        }
                                        return (suggestedSlotsData?.slots || [])
                                          .length;
                                      })()}{" "}
                                      Options
                                    </Badge>
                                  </div>
                                  <p className="text-sm text-muted-foreground leading-relaxed">
                                    We've analyzed participant feedback and
                                    availability to find the best alternative
                                    times. Review the options below and select
                                    the one that works best.
                                  </p>
                                </div>
                              </div>

                              {(() => {
                                let suggestedSlotsData =
                                  proposal.suggested_slots;
                                if (typeof suggestedSlotsData === "string") {
                                  try {
                                    suggestedSlotsData =
                                      JSON.parse(suggestedSlotsData);
                                  } catch (e) {
                                    console.error(
                                      "Failed to parse suggested_slots:",
                                      e
                                    );
                                    return null;
                                  }
                                }

                                const slots = suggestedSlotsData?.slots || [];
                                const alternatives =
                                  suggestedSlotsData?.alternatives;
                                const edgeCase = suggestedSlotsData?.edge_case;

                                if (slots.length === 0 && !alternatives) {
                                  return (
                                    <div className="p-5 bg-muted rounded-lg border border-border">
                                      <div className="flex items-start gap-4">
                                        <div className="rounded-lg bg-foreground/10 p-2.5 shrink-0">
                                          <AlertCircle className="w-6 h-6 text-foreground/70" />
                                        </div>
                                        <div className="flex-1">
                                          <h4 className="text-base font-bold text-foreground mb-2">
                                            No Available Time Slots Found
                                          </h4>
                                          <p className="text-sm text-muted-foreground mb-3 leading-relaxed">
                                            {edgeCase
                                              ? `Issue Detected: ${edgeCase}. `
                                              : ""}
                                            We couldn't find any times that work
                                            for all participants based on the
                                            current constraints.
                                          </p>
                                          <div className="bg-background rounded-lg p-3 border border-border">
                                            <p className="text-xs font-semibold text-foreground mb-1">
                                              Suggested Next Steps:
                                            </p>
                                            <ul className="text-xs text-muted-foreground space-y-1 ml-4 list-disc">
                                              <li>
                                                Provide more flexible time
                                                constraints
                                              </li>
                                              <li>
                                                Consider scheduling with fewer
                                                participants
                                              </li>
                                              <li>
                                                Contact participants directly to
                                                coordinate
                                              </li>
                                            </ul>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  );
                                }

                                return (
                                  <>
                                    {slots.length > 0 && (
                        <div className="space-y-4">
                                        {slots.map((slot, idx) => (
                                          <div
                                            key={idx}
                                            className={cn(
                                              "group relative bg-card rounded-lg overflow-hidden transition-all duration-200 hover:shadow-md",
                                              idx === 0
                                                ? "border-2 border-foreground/20"
                                                : "border border-border"
                                            )}
                                          >
                                            {/* Top Accent Bar - Only for top recommendation */}
                                            {idx === 0 && (
                                              <div className="absolute top-0 left-0 right-0 h-1 bg-foreground" />
                                            )}

                                            <div
                                              className={cn(
                                                "p-5",
                                                idx === 0 && "pt-6"
                                              )}
                                            >
                                              {/* Header Row */}
                                              <div className="flex items-center justify-between mb-4">
                                                <div className="flex items-center gap-3">
                                                  {idx === 0 ? (
                                                    <div className="flex items-center gap-2">
                                                      <div className="rounded-full bg-foreground p-2">
                                                        <svg
                                                          className="w-4 h-4 text-background"
                                                          fill="currentColor"
                                                          viewBox="0 0 20 20"
                                                        >
                                                          <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                                                        </svg>
                                                      </div>
                                                      <div>
                                                        <Badge className="bg-foreground text-background font-bold px-3 py-1 text-sm">
                                                          Best Match
                                                        </Badge>
                                                        <p className="text-xs text-muted-foreground mt-1">
                                                          Highest compatibility
                                                        </p>
                                                      </div>
                                                    </div>
                                                  ) : (
                                                    <div className="flex items-center gap-2">
                                                      <div className="rounded-full bg-muted p-2">
                                                        <CalendarIcon className="w-5 h-5 text-muted-foreground" />
                                                      </div>
                                                      <Badge
                                                        variant="secondary"
                                                        className="font-semibold px-3 py-1"
                                                      >
                                                        Option {slot.index}
                                                      </Badge>
                                                    </div>
                                                  )}
                                                </div>
                                              </div>

                                              {/* Main Content */}
                                              <div className="space-y-4">
                                                {/* Date and Time Display */}
                                                <div className="bg-muted/50 rounded-lg p-4 border border-border">
                                                  <div className="flex items-start gap-3">
                                                    <div className="rounded-lg bg-foreground/10 p-2">
                                                      <CalendarIcon className="w-5 h-5 text-foreground" />
                                                    </div>
                                                    <div className="flex-1">
                                                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
                                                        Proposed Meeting Time
                                                      </p>
                                                      <p className="text-base font-bold text-foreground leading-tight">
                                                        {formatDateRange(
                                                          slot.start,
                                                          slot.end
                                                        )}
                                                      </p>
                                                    </div>
                                                  </div>
                                                </div>

                                                {/* Reasons/Benefits */}
                                                {slot.reasons &&
                                                  slot.reasons.length > 0 && (
                                                    <div className="space-y-2">
                                                      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                                                        Why This Works
                                                      </p>
                                                      <div className="flex flex-wrap gap-2">
                                                        {slot.reasons.map(
                                                          (reason, i) => (
                                                            <div
                                                              key={i}
                                                              className="inline-flex items-center gap-1.5 text-xs font-medium bg-muted text-foreground px-3 py-1.5 rounded-md border border-border"
                                                            >
                                                              <svg
                                                                className="w-3 h-3"
                                                                fill="currentColor"
                                                                viewBox="0 0 20 20"
                                                              >
                                                                <path
                                                                  fillRule="evenodd"
                                                                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                                                                  clipRule="evenodd"
                                                                />
                                                              </svg>
                                                              {reason}
                                                            </div>
                                                          )
                                                        )}
                                                      </div>
                                                    </div>
                                                  )}

                                                {/* Action Button */}
                                                <Button
                                                  size="lg"
                                                  onClick={() =>
                                                    setRescheduleConfirmDialog({
                                                      open: true,
                                                      proposal: proposal,
                                                      slotIndex: slot.index,
                                                    })
                                                  }
                                                  className={cn(
                                                    "w-full font-semibold transition-all duration-200",
                                                    idx === 0
                                                      ? "bg-foreground text-background hover:bg-foreground/90 h-11"
                                                      : "h-10"
                                                  )}
                                                  variant={
                                                    idx === 0
                                                      ? "default"
                                                      : "outline"
                                                  }
                                                >
                                                  <CheckCircle2 className="w-4 h-4 mr-2" />
                                                  {idx === 0
                                                    ? "Select This Time"
                                                    : "Select This Option"}
                                                </Button>
                                              </div>
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    )}

                                    {/* Alternative Action */}
                                    <div className="pt-5 mt-4 border-t border-border">
                                      <div className="bg-muted/50 rounded-lg p-4 border border-border">
                                        <div className="flex items-start gap-3">
                                          <div className="rounded-full bg-foreground/10 p-2 shrink-0">
                                            <AlertCircle className="w-4 h-4 text-foreground/70" />
                                          </div>
                                          <div className="flex-1">
                                            <h4 className="text-sm font-semibold text-foreground mb-1">
                                              Need Different Options?
                                            </h4>
                                            <p className="text-xs text-muted-foreground mb-3">
                                              If none of these times work, share
                                              your preferences and we'll
                                              generate new suggestions tailored
                                              to your needs.
                                            </p>
                                            <Button
                                              variant="outline"
                                              size="default"
                                              onClick={() =>
                                                handleRejectRescheduleClick(
                                                  proposal.proposal_id
                                                )
                                              }
                                              disabled={isLoading}
                                              className="w-full font-semibold"
                                            >
                                              <RefreshCcw className="w-4 h-4 mr-2" />
                                              Provide New Constraints
                                            </Button>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  </>
                                );
                              })()}
                            </div>
                          )}

                        <div className="space-y-4">
                          <div className="flex items-center gap-2 text-base font-semibold text-foreground">
                            <div className="rounded-md bg-foreground/10 p-1.5">
                              <Users className="w-4 h-4 text-foreground" />
                            </div>
                            <span>Participant Responses</span>
                            <Badge variant="secondary" className="ml-auto">
                              {proposal.participant_responses?.filter(
                                (r) => r.response === "accepted"
                              ).length || 0}{" "}
                              / {proposal.participant_responses?.length || 0}{" "}
                              Accepted
                            </Badge>
                          </div>
                          <div className="grid gap-3">
                            {proposal.participant_responses?.map((resp) => (
                              <div
                                key={resp.participant_id}
                                className={cn(
                                  "flex flex-col gap-3 p-4 rounded-lg border transition-all",
                                  resp.response === "accepted"
                                    ? "bg-muted/50 border-border"
                                    : resp.response === "rejected"
                                    ? "bg-muted/30 border-border opacity-80"
                                    : "bg-muted/40 border-border"
                                )}
                              >
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-3">
                                    <div className="rounded-full bg-foreground/10 p-2">
                                    {resp.response === "accepted" && (
                                        <CheckCircle2 className="w-5 h-5 text-foreground" />
                                    )}
                                    {resp.response === "rejected" && (
                                        <XCircle className="w-5 h-5 text-foreground" />
                                    )}
                                    {resp.response === "pending" && (
                                        <Clock className="w-5 h-5 text-foreground" />
                                      )}
                                    </div>
                                    <div>
                                      <span className="font-semibold text-base">
                                        {resp.participant_id}
                                      </span>
                                    </div>
                                  </div>
                                    <Badge
                                    variant="secondary"
                                    className="font-semibold capitalize px-3 py-1"
                                    >
                                      {formatStatus(resp.response)}
                                    </Badge>
                                  </div>
                                {resp.response === "rejected" &&
                                  resp.feedback && (
                                    <div className="p-3 bg-muted rounded-md border border-border">
                                      <div className="flex items-start gap-2">
                                        <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                                        <div className="flex-1">
                                          <span className="font-semibold block text-xs uppercase tracking-wider mb-1">
                                            Feedback
                                    </span>
                                          <p className="text-sm">
                                            {resp.feedback}
                                          </p>
                                        </div>
                                      </div>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>

                          {/* Action Buttons Section */}
                          {proposal.participant_responses?.some(
                            (r) => r.response === "rejected"
                          ) &&
                            proposal.status === "pending" && (
                              <div className="pt-4 mt-4 border-t-2 border-dashed space-y-3">
                                <div className="flex items-center gap-2 text-sm font-semibold text-muted-foreground mb-3">
                                  <AlertCircle className="w-4 h-4" />
                                  <span>Action Required</span>
                                </div>

                                <div className="flex flex-wrap gap-3">
                              <Button
                                    variant="outline"
                                    className="flex-1 sm:flex-none gap-2 font-semibold"
                                onClick={() => handleReschedule(proposal)}
                                disabled={isLoading}
                              >
                                <RefreshCcw
                                  className={cn(
                                    "w-4 h-4",
                                    isLoading && "animate-spin"
                                  )}
                                />
                                    Get Alternative Times
                              </Button>
                              
                                  {calculateMajority(
                                    proposal.participant_responses,
                                    proposal.priority_participants
                                  ) ? (
                                <Button
                                      className="flex-1 sm:flex-none gap-2 font-semibold"
                                      onClick={() =>
                                        handlePushThrough(proposal)
                                      }
                                  disabled={isLoading}
                                >
                                      <CheckCircle2 className="w-4 h-4" />
                                      Finalize (Majority Approved)
                                </Button>
                              ) : (
                                proposal.participant_responses?.some(
                                  (r) => 
                                    r.response === "rejected" && 
                                        proposal.priority_participants?.includes(
                                          r.participant_id
                                        )
                                    ) && (
                                      <div className="flex items-start gap-2 text-sm bg-muted px-4 py-3 rounded-lg border border-border flex-1">
                                        <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                                        <div>
                                          <p className="font-semibold mb-1">
                                            Cannot Finalize
                                          </p>
                                          <p className="text-xs text-muted-foreground">
                                            A priority participant has declined.
                                            Please reschedule or modify the
                                            meeting.
                                          </p>
                                        </div>
                                  </div>
                                )
                              )}
                                </div>
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
                <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-full bg-muted">
                  <CheckCircle2 className="w-8 h-8 text-foreground" />
                </div>
                <DialogTitle className="text-center text-xl">
                  {confirmationDialog.meetingData?.isReschedule
                    ? "Reschedule Confirmed!"
                    : "Meeting Confirmed!"}
                </DialogTitle>
                <DialogDescription className="text-center">
                  {confirmationDialog.meetingData?.isReschedule
                    ? "The meeting has been rescheduled and participants notified."
                    : "The meeting has been successfully scheduled."}
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
                    <CalendarIcon className="w-4 h-4 text-foreground" />
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
                    <Users className="w-4 h-4 text-foreground" />
                    <h3 className="font-semibold text-sm text-muted-foreground">
                      {confirmationDialog.meetingData?.isReschedule
                        ? "Notified Participants"
                        : "Confirmed Attendees"}
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
                      {confirmationDialog.meetingData?.isReschedule
                        ? "participants notified"
                        : "participants confirmed"}
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

      {/* Reschedule Confirmation Dialog */}
      <Dialog
        open={rescheduleConfirmDialog.open}
        onOpenChange={(open) =>
          !open &&
          setRescheduleConfirmDialog({
            open: false,
            proposal: null,
            slotIndex: null,
          })
        }
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-full bg-muted">
              <RefreshCcw className="w-7 h-7 text-foreground" />
            </div>
            <DialogTitle className="text-center text-xl">
              Confirm Reschedule
            </DialogTitle>
            <DialogDescription className="text-center">
              Ready to reschedule? All participants will be notified of the new
              time and need to accept the update.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4 px-2">
            <div className="p-4 bg-muted/50 rounded-lg border border-border">
              <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                <CalendarIcon className="w-4 h-4" />
                <span className="font-semibold">New Meeting Time</span>
              </div>
              <p className="text-base font-bold text-foreground">
                {rescheduleConfirmDialog.proposal &&
                  rescheduleConfirmDialog.slotIndex &&
                  (() => {
                    let suggestedSlotsData =
                      rescheduleConfirmDialog.proposal.suggested_slots;
                    if (typeof suggestedSlotsData === "string") {
                      try {
                        suggestedSlotsData = JSON.parse(suggestedSlotsData);
                      } catch (e) {
                        return "Time slot details unavailable";
                      }
                    }
                    const slots = suggestedSlotsData?.slots || [];
                    const selectedSlot = slots.find(
                      (s) => s.index === rescheduleConfirmDialog.slotIndex
                    );
                    return selectedSlot
                      ? formatDateRange(selectedSlot.start, selectedSlot.end)
                      : "Time slot details unavailable";
                  })()}
              </p>
            </div>
          </div>
          <DialogFooter className="gap-2">
            <Button
              variant="outline"
              onClick={() =>
                setRescheduleConfirmDialog({
                  open: false,
                  proposal: null,
                  slotIndex: null,
                })
              }
              className="flex-1"
            >
              Cancel
            </Button>
            <Button
              onClick={() =>
                handleConfirmReschedule(
                  rescheduleConfirmDialog.proposal,
                  rescheduleConfirmDialog.slotIndex
                )
              }
              disabled={isLoading}
              className="flex-1"
            >
              {isLoading ? (
                <>
                  <RefreshCcw className="w-4 h-4 mr-2 animate-spin" />
                  Confirming...
                </>
              ) : (
                <>
                  <CheckCircle2 className="w-4 h-4 mr-2" />
                  Confirm Reschedule
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Reschedule Rejection Dialog */}
      <Dialog
        open={rescheduleRejectDialog.open}
        onOpenChange={(open) =>
          !open && setRescheduleRejectDialog({ open: false, proposalId: null })
        }
      >
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 rounded-full bg-muted">
              <AlertCircle className="w-7 h-7 text-foreground" />
            </div>
            <DialogTitle className="text-center text-xl">
              Need Different Options?
            </DialogTitle>
            <DialogDescription className="text-center">
              Let us know your preferences and constraints. We'll generate new
              suggestions tailored to your needs.
            </DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <Label
              htmlFor="organizer-feedback"
              className="mb-3 block text-sm font-semibold"
            >
              Your Constraints & Preferences
            </Label>
            <textarea
              id="organizer-feedback"
              placeholder="Examples:&#10;• I prefer mornings (9-11 AM)&#10;• Need it next week instead&#10;• Can't do Mondays or Fridays&#10;• Earlier in the day works better"
              value={organizerFeedback}
              onChange={(e) => setOrganizerFeedback(e.target.value)}
              className="flex min-h-[140px] w-full rounded-lg border border-input bg-background px-4 py-3 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none"
            />
            <p className="text-xs text-muted-foreground mt-2">
              Be specific to get better suggestions
            </p>
          </div>
          <DialogFooter className="gap-2">
            <Button
              variant="outline"
              onClick={() =>
                setRescheduleRejectDialog({ open: false, proposalId: null })
              }
              className="flex-1"
            >
              Cancel
            </Button>
            <Button
              onClick={confirmRejectReschedule}
              disabled={isLoading || !organizerFeedback.trim()}
              className="flex-1"
            >
              {isLoading ? (
                <>
                  <RefreshCcw className="w-4 h-4 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <RefreshCcw className="w-4 h-4 mr-2" />
                  Generate New Suggestions
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Dialog>
  );
}
