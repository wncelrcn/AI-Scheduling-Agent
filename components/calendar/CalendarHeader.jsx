import React from "react";
import { format } from "date-fns";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

export function CalendarHeader({ currentDate, onPrevMonth, onNextMonth, onToday }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <h2 className="text-2xl font-bold text-foreground">
        {format(currentDate, "MMMM yyyy")}
      </h2>
      <div className="flex items-center gap-2">
        <div className="flex items-center border rounded-md bg-background">
          <Button
            variant="ghost"
            size="icon"
            onClick={onPrevMonth}
            aria-label="Previous month"
            className="h-8 w-8 rounded-none rounded-l-md"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            onClick={onToday}
            className="h-8 px-3 rounded-none border-x text-sm font-medium"
          >
            Today
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={onNextMonth}
            aria-label="Next month"
            className="h-8 w-8 rounded-none rounded-r-md"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

