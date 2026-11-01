
# Handling Challenging Scenarios

## 1. Occlusion

Occlusion occurs when a person is temporarily hidden by another person, shelves, displays or other objects.

**Approach used:**
- The tracker uses motion prediction to estimate a person's next position.
- New detections are matched with existing tracks using bounding-box overlap and motion information.
- Two-stage detection matching helps maintain tracks when detection confidence is lower.

**Limitations:**
- Long-duration or heavy occlusion can cause tracking IDs to change.
- A person who disappears completely may be treated as a new track when detected again.

**Future improvement:**
- Use a stronger tracking algorithm with appearance-based Re-Identification (Re-ID).

## 2. Groups Entering Together

When multiple people enter together, their bounding boxes may overlap.

**Approach used:**
- The YOLOv8n detector attempts to detect each person separately.
- The tracker assigns individual IDs to detected people.
- Each confirmed track crossing the entrance line is evaluated as a separate event.

**Limitations:**
- People standing very close together may be detected as one person or missed.
- Overlapping bounding boxes can cause incorrect tracking assignments.

**Future improvement:**
- Use a more accurate person detector and evaluate group-entry scenarios with additional annotated footage.

## 3. People Re-entering

A person may leave the store and return later.

**Approach used:**
- IN and OUT events are determined by the direction in which a tracked person crosses the entrance line.
- A new entry after an exit can generate another IN event.

**Limitations:**
- Tracking IDs are temporary identifiers, not permanent identities.
- A returning person may receive a new ID.
- The system does not identify whether two different tracks belong to the same individual.

**Future improvement:**
- Introduce person Re-ID if permitted, with appropriate privacy safeguards.

## 4. Staff vs. Customers

**Approach used:**
- The current system detects and counts all people equally.
- It does not distinguish employees from customers.

**Limitations:**
- Staff entering or leaving can affect the customer occupancy count.
- The system cannot independently determine a person's role from the current detection and tracking pipeline.

**Future improvement:**
- Introduce an explicitly labelled staff classification or an authorized staff-exclusion mechanism, subject to privacy and operational requirements.

## Conclusion

The system provides an offline person detection, tracking and entrance/exit counting pipeline. It handles basic movement and temporary tracking challenges, but heavy occlusion, overlapping groups, re-entry identity and staff/customer separation remain limitations that require further development.
