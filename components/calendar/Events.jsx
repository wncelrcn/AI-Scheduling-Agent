import React from "react";
import { isSameDay } from "date-fns";

export function Events({ currentDate, children }) {
  // Dummy events data
  const events = [
    {
      date: new Date(currentDate.getFullYear(), currentDate.getMonth(), 5),
      title: "Weekly Meeting",
      variant: "blue",
    },
    {
      date: new Date(currentDate.getFullYear(), currentDate.getMonth(), 12),
      title: "Weekly Meeting",
      variant: "blue",
    },
    {
      date: new Date(currentDate.getFullYear(), currentDate.getMonth(), 20),
      title: "Review Q3 Goals",
      variant: "red",
    },
    {
      date: new Date(currentDate.getFullYear(), currentDate.getMonth(), 2),
      title: "Week 12 (Week 7)",
      variant: "green",
    },
    {
      date: new Date(currentDate.getFullYear(), currentDate.getMonth(), 9),
      title: "Week 13",
      variant: "green",
    },
    {
      date: new Date(currentDate.getFullYear(), currentDate.getMonth(), 16),
      title: "Week 14",
      variant: "green",
    },
  ];

  const getEventsForDay = (day) => {
    return events.filter((event) => isSameDay(event.date, day));
  };

  return children({ events, getEventsForDay });
}

