"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  addMonths,
  subMonths,
  startOfMonth,
  endOfMonth,
  startOfWeek,
  endOfWeek,
  eachDayOfInterval,
  isSameMonth,
  isToday,
} from "date-fns";
import { CalendarHeader } from "./CalendarHeader";
import { CalendarDay } from "./CalendarDay";
import { Events } from "./Events";

export function Calendar({ username }) {
  const [currentDate, setCurrentDate] = useState(new Date());
  const todayRef = useRef(null);
  const gridRef = useRef(null);

  // Navigation handlers
  const prevMonth = () => setCurrentDate(subMonths(currentDate, 1));
  const nextMonth = () => setCurrentDate(addMonths(currentDate, 1));
  const goToToday = () => setCurrentDate(new Date());

  // Generate calendar grid
  const monthStart = startOfMonth(currentDate);
  const monthEnd = endOfMonth(monthStart);
  const startDate = startOfWeek(monthStart, { weekStartsOn: 1 }); // Monday start
  const endDate = endOfWeek(monthEnd, { weekStartsOn: 1 });

  const calendarDays = eachDayOfInterval({
    start: startDate,
    end: endDate,
  });

  const weekDays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  const weeks = Math.ceil(calendarDays.length / 7);

  // Scroll to today's date on mount and when currentDate changes
  useEffect(() => {
    // Use setTimeout to ensure DOM is fully rendered
    const timer = setTimeout(() => {
      if (todayRef.current && gridRef.current) {
        const gridElement = gridRef.current;
        const todayElement = todayRef.current;
        
        // Get the offset position of today's element relative to the grid
        const todayOffsetTop = todayElement.offsetTop;
        const gridScrollTop = gridElement.scrollTop;
        const gridClientHeight = gridElement.clientHeight;
        
        // Calculate scroll position to center today's date
        const scrollPosition = todayOffsetTop - gridClientHeight / 2 + todayElement.offsetHeight / 2;
        
        gridElement.scrollTo({
          top: scrollPosition,
          behavior: 'smooth'
        });
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [currentDate]);

  return (
    <div className="flex flex-col h-full w-full max-w-7xl mx-auto bg-background rounded-lg border shadow-sm p-2 sm:p-4">
      <CalendarHeader
        currentDate={currentDate}
        onPrevMonth={prevMonth}
        onNextMonth={nextMonth}
        onToday={goToToday}
      />

      <Events currentDate={currentDate} username={username}>
        {({ getEventsForDay }) => (
          <div className="flex flex-col border rounded-lg overflow-hidden bg-muted flex-1 min-h-0">
            {/* Sticky weekday headers */}
            <div className="grid grid-cols-7 gap-px bg-muted border-b sticky top-0 z-10">
              {weekDays.map((day) => (
                <div
                  key={day}
                  className="bg-background p-2 text-center text-sm font-medium text-muted-foreground"
                >
                  {day}
                </div>
              ))}
            </div>

            {/* Scrollable calendar days */}
            <div
              ref={gridRef}
              className="grid grid-cols-7 gap-px bg-muted overflow-y-auto overflow-x-hidden flex-1 min-h-0"
              style={{ gridTemplateRows: `repeat(${weeks}, minmax(120px, auto))` }}
            >
              {calendarDays.map((day, index) => (
                <div
                  key={day.toISOString()}
                  ref={isToday(day) ? todayRef : null}
                >
                  <CalendarDay
                    date={day}
                    isCurrentMonth={isSameMonth(day, monthStart)}
                    isToday={isToday(day)}
                    events={getEventsForDay(day)}
                    className="bg-background min-h-0"
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </Events>
    </div>
  );
}

