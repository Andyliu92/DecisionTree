#!/usr/bin/env python3

from redmail.email.sender import EmailSender
import argparse

email = EmailSender(
    host='smtp-mail.outlook.com',
    port=587,
    username='andyliu.pub@outlook.com',
    password='0902Lsy_123312'
)
parser = argparse.ArgumentParser(description='test')
parser.add_argument('-m',default="Completed task!", type=str, help ='Message in the email')
args = parser.parse_args()

email.send(
    subject=args.m,
    sender="comfluter@outlook.com",
    receivers=['andyliu.nju@outlook.com'],
    text=args.m,
    html="<h2>"+ args.m +"</h2>"
)