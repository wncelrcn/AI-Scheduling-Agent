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
            organizer_id,
            reasoning
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
      const { error } = await supabase
        .from("participant_responses")
        .update({
          response: "rejected",
          feedback: rejectReason,
        })
        .eq("proposal_id", rejectDialog.proposalId)
        .eq("participant_id", username);

      if (error) throw error;

      setRejectDialog({ open: false, proposalId: null });
      fetchProposals();
    } catch (error) {
      console.error("Error rejecting proposal:", error);
    }
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
            <DialogDescription>
              Manage your meeting proposals and invitations.
            </DialogDescription>
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
              <div className="space-y-4">
                {incomingProposals.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <p>No incoming meeting requests.</p>
                  </div>
                ) : (
                  incomingProposals.map((proposal) => (
                    <Card
                      key={proposal.proposal_id}
                      className="overflow-hidden border-l-4 border-l-primary"
                    >
                      <CardHeader className="pb-3">
                        <div className="flex justify-between items-start">
                          <div>
                            <CardTitle className="text-lg">
                              {proposal.meeting_title || "Untitled Meeting"}
                            </CardTitle>
                            <CardDescription>
                              Organized by {proposal.organizer_id}
                            </CardDescription>
                          </div>
                          <Badge
                            variant={getStatusBadgeVariant(proposal.my_status)}
                            className={cn(
                              "capitalize",
                              getStatusColorClass(proposal.my_status)
                            )}
                          >
                            {formatStatus(proposal.my_status)}
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent className="pb-3 text-sm grid gap-2">
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <CalendarIcon className="w-4 h-4" />
                          <span>
                            {formatDate(proposal.proposed_start)} -{" "}
                            {formatDate(proposal.proposed_end)}
                          </span>
                        </div>
                        {proposal.reasoning && (
                          <div className="bg-muted p-3 rounded-md text-sm mt-2 flex gap-2">
                            <AlertCircle className="w-4 h-4 text-muted-foreground shrink-0 mt-0.5" />
                            <div>
                              <span className="font-semibold block text-xs uppercase tracking-wider text-muted-foreground mb-1">
                                Organizer Reasoning
                              </span>
                              {proposal.reasoning}
                            </div>
                          </div>
                        )}
                      </CardContent>
                      <CardFooter className="bg-muted/50 p-3 flex justify-end gap-2">
                        {proposal.my_status === "pending" && (
                          <>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() =>
                                handleRejectClick(proposal.proposal_id)
                              }
                              className="text-destructive hover:text-destructive hover:bg-destructive/10 border-destructive/20"
                            >
                              Reject / Negotiate
                            </Button>
                            <Button
                              size="sm"
                              onClick={() => handleAccept(proposal.proposal_id)}
                              className="bg-green-600 hover:bg-green-700"
                            >
                              Accept
                            </Button>
                          </>
                        )}
                        {proposal.my_status !== "pending" && (
                          <span className="text-xs text-muted-foreground italic">
                            You have {proposal.my_status} this proposal.
                          </span>
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
                              {formatDate(proposal.proposed_start)}
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
                                {resp.response === "rejected" &&
                                  resp.feedback && (
                                    <div className="text-sm bg-destructive/10 text-destructive p-2 rounded border border-destructive/20">
                                      <span className="font-semibold block text-xs mb-1">
                                        Reason for Rejection:
                                      </span>
                                      {resp.feedback}
                                    </div>
                                  )}
                              </div>
                            ))}
                          </div>

                          {/* Re-generate Schedule Button if any rejection */}
                          {proposal.participant_responses?.some(
                            (r) => r.response === "rejected"
                          ) && (
                            <div className="pt-2">
                              <Button
                                className="w-full sm:w-auto gap-2"
                                variant="secondary"
                              >
                                <RefreshCcw className="w-4 h-4" />
                                Re-generate Schedule with Feedback
                              </Button>
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
    </Dialog>
  );
}
