# UltrasoundPig API

## ภาพรวม (Overview)

UltrasoundPig คือ API ที่พัฒนาด้วย FastAPI (Python) และทำงานบน Docker มีหน้าที่หลักในการรับไฟล์ PDF (ภาพอัลตราซาวด์) จากผู้ใช้งาน, แปลงไฟล์เป็นรูปภาพ PNG, ดึงข้อมูลจากภาพ (OCR), และบันทึกผลลัพธ์ลงในฐานข้อมูล MySQL

โปรเจคนี้ถูกออกแบบมาให้ง่ายต่อการติดตั้งและใช้งานผ่าน Docker, มีระบบจัดการ Log อัตโนมัติ, และมีเอกสารประกอบที่ชัดเจน

## ฟีเจอร์หลัก (Key Features)

- **PDF Processing**: รับไฟล์ PDF และแปลงเป็นรูปภาพ PNG
- **OCR Integration**: ดึงข้อมูลตัวอักษรจากรูปภาพ
- **Database Storage**: บันทึกข้อมูลลงใน MySQL
- **Dockerized**: รันโปรเจคทั้งหมดได้ง่ายๆ ผ่าน Docker และ Docker Compose
- **Configurable**: ตั้งค่าการเชื่อมต่อ, AI model, threshold, และ dry-run DB ผ่านไฟล์ `config/.env`
- **Automated Log Management**: มีระบบหมุนเวียนไฟล์ Log (Log Rotation) อัตโนมัติ ป้องกันปัญหาไฟล์ Log ใหญ่เกินไป

## การเริ่มต้นใช้งาน

สำหรับนักพัฒนาที่ต้องการรันโปรเจคนี้ในเครื่องของตัวเอง กรุณาทำตามขั้นตอนในเอกสาร:

- **ใช้ `config/.env` เป็น config จริงเสมอ** ส่วน `config/.env.example` เป็น template/mockup ให้รู้ว่าต้องตั้งค่าอะไรบ้าง
- **[คู่มือการใช้งาน Docker (README-Docker.md)](./README-Docker.md)**: สำหรับการรันโปรเจคด้วย Docker ซึ่งเป็นวิธีที่แนะนำ

## เอกสารสำหรับนักพัฒนาและผู้ดูแลระบบ

สำหรับข้อมูลเกี่ยวกับโครงสร้างโปรเจค, Docker, runtime config, smoke test, และ deployment ให้ใช้เอกสารหลักชุดนี้:

- **[README-Docker.md](./README-Docker.md)**: คู่มือรันระบบด้วย Docker, config จริง, smoke test, และคำสั่ง deployment ที่ใช้ได้จริง
- **[Documentation/README.md](./Documentation/README.md)**: แผนที่เอกสารย่อยและขอบเขตว่าไฟล์ไหนเป็น reference เรื่องอะไร
- **[AGENTS.MD](./AGENTS.MD)**: กติกาการทำงานของ coding agent ใน repo นี้ ไม่ใช่คู่มือผู้ใช้ระบบ
