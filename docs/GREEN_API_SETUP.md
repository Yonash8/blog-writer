# Green API: Connect Your Number and Get API Keys

Follow these steps to connect WhatsApp to your blog-writing agent via Green API.

## Step 1: Install WhatsApp on your phone

- Install [WhatsApp](https://www.whatsapp.com/) or [WhatsApp Business](https://www.whatsapp.com/business/) on your mobile phone.
- Your phone must be charged and connected to the internet when the bot is running, since messages are sent/received through it.
- **Tip:** Consider Green API's "Handset hosting" service if you do not want to use a personal phone.

## Step 2: Register in the Green API console

1. Go to [console.green-api.com](https://console.green-api.com).
2. Enter your email (prefer a corporate email).
3. Select your country.
4. Accept the user agreement and click **Register**.
5. Check your inbox (and Spam) for a confirmation code.
6. Enter the code to verify your account.

## Step 3: Create an instance

1. In the console sidebar, click **Instances**.
2. Click **Create an instance**.
3. Choose a plan and pay if required.
4. Wait up to 2 minutes for the instance to become operational.

## Step 4: Connect your WhatsApp number (authorize the instance)

1. On your phone, open WhatsApp or WhatsApp Business.
2. Go to **Link a device**:
   - **Android:** Menu (three dots) → Linked Devices → Link a device
   - **iPhone:** Settings → Linked Devices → Link a device
3. In the Green API console, select your instance.
4. Click **Get QR**.
5. Scan the QR code with your phone camera / WhatsApp scanner.

Once linked, your instance shows as authorized.

## Step 5: Copy your API credentials

In the Green API console, open your instance and locate:

| Parameter | Description | Use in app |
|-----------|-------------|------------|
| **idInstance** | Instance ID (numeric) | `GREEN_API_INSTANCE_ID` |
| **apiTokenInstance** | Instance access key | `GREEN_API_TOKEN` |
| **apiUrl** | API host (usually `https://api.green-api.com`) | `GREEN_API_BASE_URL` |

Add them to your `.env`:

```
GREEN_API_INSTANCE_ID=1101000001
GREEN_API_TOKEN=d75b3a66374942c5b3c019c698abc2067e151558acbd412345
GREEN_API_BASE_URL=https://api.green-api.com
```

**Security:** If the token is compromised, change it in the console or contact [Green API support](https://wa.me/77780739095).

## Step 6: Configure the webhook

Your app must be reachable at a public URL so Green API can send incoming messages.

**If deployed (e.g. Railway, Render, Fly.io):**
- Webhook URL: `https://your-domain.com/webhooks/green-api/whatsapp`

**If running locally:**
- Expose your server with a tunnel (ngrok, Cloudflare Tunnel).
- Example with ngrok: `ngrok http 8000` → use the `https://...` URL as the webhook base.
- Full webhook: `https://abc123.ngrok.io/webhooks/green-api/whatsapp`

**Configure in the console:**
1. Open your instance in the Green API console.
2. Find the webhook / notifications section.
3. Set **Webhook URL** to your full endpoint.
4. Enable **Receive notifications about incoming messages and files** (`incomingWebhook`).

**Or via API (SetSettings):**

```bash
curl -X POST "https://api.green-api.com/waInstance{YOUR_ID}/setSettings/{YOUR_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "webhookUrl": "https://your-domain.com/webhooks/green-api/whatsapp",
    "webhookUrlToken": "",
    "incomingWebhook": "yes"
  }'
```

Replace `YOUR_ID`, `YOUR_TOKEN`, and `webhookUrl` with your values.

## Step 7: Verify it works

1. Start your FastAPI app.
2. Send a WhatsApp message to the linked number (e.g. "Write an article about AI").
3. The agent should respond via WhatsApp.

**Debugging:**
- Use [Green API console API tester](https://console.green-api.com/app/api/) to test SendMessage.
- Use [Webhook.Site](https://webhook.site) to capture raw webhooks and confirm Green API is calling your URL.
- Ensure Green API can reach your webhook (no firewall blocking, valid HTTPS).
