from email.message import EmailMessage
from email.utils import formatdate, make_msgid


def build_message(sender: str, recipient: str, subject: str, text_body: str, html_body: str,
                  chart_png: bytes | None, chart_cid: str | None) -> EmailMessage:
    """Plain-text part, HTML alternative, and the chart once, inline in the HTML part by content id.

    No separate attached copy: mail apps showed the chart twice. The plain-text part has the metrics table and no
    chart."""
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, recipient, subject
    msg["Date"], msg["Message-ID"] = formatdate(localtime=True), make_msgid()
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    if chart_png and chart_cid:
        html_part = msg.get_payload()[1]
        html_part.add_related(chart_png, maintype="image", subtype="png", cid=f"<{chart_cid}>", filename="chart.png", disposition="inline")
    return msg
