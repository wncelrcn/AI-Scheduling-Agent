import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { format } from "date-fns";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export function CalendarDay({ date, isCurrentMonth, isToday, events = [], className }) {
  const [isOpen, setIsOpen] = useState(false);
  const maxVisibleEvents = 3;
  const hasMoreEvents = events.length > maxVisibleEvents;
  
  // Show maxVisibleEvents items, and if there are more, add the "+X more" item below
  const displayLimit = maxVisibleEvents;
  const visibleEvents = events.slice(0, displayLimit);

  const EventItem = ({ event, isCompact = true }) => (
    <div
      className={cn(
        "text-xs rounded-md truncate font-medium border transition-all hover:opacity-80",
        isCompact ? "px-1.5 py-0.5" : "px-3 py-2",
        event.variant === "blue" && "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-900/30 dark:text-blue-300 dark:border-blue-800",
        event.variant === "green" && "bg-green-100 text-green-700 border-green-200 dark:bg-green-900/30 dark:text-green-300 dark:border-green-800",
        event.variant === "red" && "bg-red-100 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-300 dark:border-red-800",
        !event.variant && "bg-primary/10 text-primary border-primary/20"
      )}
      title={`${event.title} (${event.timeRange})`}
    >
      <div className={cn("font-semibold truncate", !isCompact && "text-sm")}>{event.title}</div>
      <div className={cn("opacity-90 truncate", isCompact ? "text-[10px]" : "text-xs mt-0.5")}>{event.timeRange}</div>
    </div>
  );

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <div
        className={cn(
          "group min-h-[120px] sm:min-h-[180px] h-full bg-background p-1 sm:p-2 flex flex-col gap-1 transition-colors hover:bg-accent/5 relative cursor-pointer",
          !isCurrentMonth && "bg-muted/10 text-muted-foreground",
          isToday && "bg-accent/10",
          className
        )}
        onClick={() => setIsOpen(true)}
        data-testid="calendar-day"
      >
        <div className="flex justify-between items-start">
          <span
            className={cn(
              "text-sm font-medium w-7 h-7 flex items-center justify-center rounded-full transition-all",
              isToday 
                ? "bg-primary text-primary-foreground shadow-sm" 
                : "text-muted-foreground group-hover:text-foreground group-hover:bg-muted"
            )}
          >
            {format(date, "d")}
          </span>
        </div>
        
        <div className="flex flex-col gap-1 mt-1 flex-1">
          {visibleEvents.map((event, index) => (
             <EventItem key={index} event={event} />
          ))}
          
          {hasMoreEvents && (
            <div
                className="text-[10px] font-medium text-muted-foreground hover:text-foreground hover:bg-muted/80 px-1.5 py-0.5 rounded w-full text-left transition-colors"
            >
                +{events.length - displayLimit} more
            </div>
          )}
        </div>
      </div>

      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span className="text-xl font-bold">{format(date, "EEEE, MMMM d")}</span>
            <span className="text-sm font-normal text-muted-foreground">({events.length} events)</span>
          </DialogTitle>
        </DialogHeader>
        <div className="grid gap-2 py-4 max-h-[60vh] overflow-y-auto pr-2">
          {events.length > 0 ? events.map((event, index) => (
             <div key={index} className="flex flex-col gap-1">
                 <EventItem event={event} isCompact={false} />
             </div>
          )) : (
            <div className="text-center text-muted-foreground py-8">No events scheduled for this day.</div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
