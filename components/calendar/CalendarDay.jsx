import React from "react";
import { cn } from "@/lib/utils";
import { format } from "date-fns";

export function CalendarDay({ date, isCurrentMonth, isToday, events = [], className }) {
  return (
    <div
      className={cn(
        "min-h-[120px] sm:min-h-[140px] border p-1 sm:p-2 flex flex-col gap-1 transition-colors hover:bg-muted/50",
        !isCurrentMonth && "bg-muted/20 text-muted-foreground",
        isToday && "bg-accent/10",
        className
      )}
      data-testid="calendar-day"
    >
      <div className="flex justify-end items-start">
        <span
          className={cn(
            "text-sm font-medium w-7 h-7 flex items-center justify-center rounded-full",
            isToday && "bg-primary text-primary-foreground"
          )}
        >
          {format(date, "d")}
        </span>
      </div>
      
      <div className="flex flex-col gap-1 mt-1">
        {events.map((event, index) => (
          <div
            key={index}
            className={cn(
              "text-xs p-1 rounded truncate font-medium",
              event.variant === "blue" && "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300",
              event.variant === "green" && "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
              event.variant === "red" && "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
              !event.variant && "bg-primary/10 text-primary"
            )}
            title={event.title}
          >
            <div className="font-semibold">{event.title}</div>
            <div className="text-[10px] opacity-90">{event.timeRange}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

