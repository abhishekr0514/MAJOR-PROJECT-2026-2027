In the **current version** we built: **Yes**, the Super Admin sets it.

However, in a **high-end production system**, the Super Admin should **not** know the Hospital Admin's password. Here is how that is usually handled:

### 1. The "Temporary Password" Pattern
The Super Admin sets a random password, and the user is forced to change it on their first login. 
*   *Pros*: Easy to implement.
*   *Cons*: Slightly less secure (the password travels through the Super Admin).

### 2. The "Invitation Flow" (The Professional Way)
1.  Super Admin only provides the **Email** and **Name**.
2.  The backend creates the user with no password (or a disabled state).
3.  The backend generates a **Secure Invitation Token** and sends an email to the Hospital Admin.
4.  The Hospital Admin clicks the link and chooses their own password.

### Which should we use?
If you want to keep the "Very High Quality" standard we started with, we should move toward the **Invitation Flow**. 

**Since we don't have an email server set up yet**, a good middle-ground is:
*   The API returns a **Registration Link** (containing a one-time token) when the Super Admin creates the user.
*   The Super Admin can copy that link and send it to the Hospital Admin (via WhatsApp/Email/Slack).

**Would you like me to implement this "Token-based Set Password" logic?** It would remove the `password` field from the Admin's creation form and make the system much more professional.