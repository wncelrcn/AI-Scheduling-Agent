import React, { useEffect, useState } from "react";
import { isSameDay, isWithinInterval, startOfDay, endOfDay, format } from "date-fns";
import { createClient } from "@/utils/supabase/client";

export function Events({ currentDate, username, children }) {
  const [events, setEvents] = useState([]);
  const supabase = createClient();

  useEffect(() => {
    const fetchEvents = async () => {
      if (!username) return;

      const { data, error } = await supabase
        .from("meetings")
        .select("*")
        .eq("user", username);

      if (error) {
        console.error("Error fetching events:", error);
        return;
      }

      if (data) {
        const formattedEvents = data.map((event) => {
          const startDate = new Date(event.start_meeting);
          const endDate = new Date(event.end_meeting);
          return {
            start: startDate,
            end: endDate,
            title: event.meeting_name,
            variant: "blue", // Default color
            timeRange: `${format(startDate, "h:mm a")} - ${format(endDate, "h:mm a")}`,
          };
        });
        setEvents(formattedEvents);
      }
    };

    fetchEvents();
  }, [username]);

  const getEventsForDay = (day) => {
    return events.filter((event) => {
      const dayStart = startOfDay(day);
      const dayEnd = endOfDay(day);
      
      // Check if the event overlaps with the current day
      return (
        isWithinInterval(dayStart, { start: startOfDay(event.start), end: endOfDay(event.end) }) ||
        isWithinInterval(dayEnd, { start: startOfDay(event.start), end: endOfDay(event.end) }) ||
        (event.start < dayStart && event.end > dayEnd)
      );
    });
  };

  return children({ events, getEventsForDay });
}
