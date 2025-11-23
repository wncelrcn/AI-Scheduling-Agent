import React from "react";
import { format } from "date-fns";
import { ChevronLeft, ChevronRight, RefreshCcw, Calendar as CalendarIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function CalendarHeader({
  currentDate,
  onPrevMonth,
  onNextMonth,
  onToday,
  onRefresh,
  isLoading,
}) {
  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 gap-4">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary/10 rounded-lg hidden sm:block">
          <CalendarIcon className="w-6 h-6 text-primary" />
        </div>
        <div>
            <h2 className="text-2xl font-bold text-foreground leading-none tracking-tight">
                {format(currentDate, "MMMM yyyy")}
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
                View and see your upcoming meetings
            </p>
        </div>
      </div>
      
      <div className="flex items-center gap-2 w-full sm:w-auto justify-between sm:justify-end">
        <Button
          variant="outline"
          size="sm"
          onClick={onRefresh}
          disabled={isLoading}
          className="h-9 w-9 p-0 shrink-0"
          title="Refresh events"
        >
          <RefreshCcw className={cn("h-4 w-4", isLoading && "animate-spin")} />
          <span className="sr-only">Refresh</span>
        </Button>
        
        <div className="flex items-center rounded-md border bg-background shadow-sm">
          <Button
            variant="ghost"
            size="sm"
            onClick={onPrevMonth}
            className="h-9 w-9 p-0 rounded-none rounded-l-md hover:bg-muted"
          >
            <ChevronLeft className="h-4 w-4" />
             <span className="sr-only">Previous month</span>
          </Button>
          <div className="w-px h-4 bg-border" />
          <Button
            variant="ghost"
            size="sm"
            onClick={onToday}
            className="h-9 px-4 rounded-none font-medium hover:bg-muted"
          >
            Today
          </Button>
          <div className="w-px h-4 bg-border" />
          <Button
            variant="ghost"
            size="sm"
            onClick={onNextMonth}
            className="h-9 w-9 p-0 rounded-none rounded-r-md hover:bg-muted"
          >
            <ChevronRight className="h-4 w-4" />
             <span className="sr-only">Next month</span>
          </Button>
        </div>
      </div>
    </div>
  );
}
