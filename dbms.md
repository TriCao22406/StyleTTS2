**Kịch Bản Case Study: "Bí Ẩn Dữ Liệu Biến Mất"**

---

### **Bối Cảnh**  
Công ty **SecureTech** chuyên về an ninh mạng, sở hữu cơ sở dữ liệu (CSDL) cực kỳ nhạy cảm chứa thông tin khách hàng và mã nguồn dự án. Để đảm bảo an toàn, công ty triển khai hệ thống giám sát hoạt động người dùng (theo Practice4) và cơ chế sao lưu tự động (theo Practice6). Tuy nhiên, một sự cố bất ngờ xảy ra: dữ liệu quan trọng trong bảng `SensitiveData` bị xóa hàng loạt vào lúc nửa đêm. Admin phải điều tra và khôi phục dữ liệu trước khi công ty chịu tổn thất lớn.

---

### **Diễn Biến**  

#### **Giai Đoạn 1: Thiết Lập Hệ Thống**  
1. **AuditLog & Trigger**  
   - Admin tạo bảng `SensitiveData` và `AuditLog` để ghi lại mọi truy vấn.  
   - Triển khai **Trigger `trg_AuditSensitiveData_Full_SQL`** để ghi nhận INSERT/UPDATE/DELETE, kèm lệnh SQL thực thi.  
   ```sql
   CREATE TRIGGER trg_AuditSensitiveData_Full_SQL
   ON dbo.SensitiveData
   AFTER INSERT, UPDATE, DELETE
   AS
   BEGIN
       -- Logic ghi log vào AuditLog (đã có trong Practice4)
   END;
   ```

2. **Phân Quyền & Backup Tự Động**  
   - Tạo user **UserX1** (nhân viên mới) với quyền `db_datawriter` và `db_datareader`.  
   - Thiết lập **Job Backup 5 Giây** (Practice6) để sao lưu CSDL `DemoDB` vào `C:\Backup` với tên file có timestamp.  
   ```powershell
   while ($true) {
       $timestamp = Get-Date -Format "ddMMyyyy_HHmmss";
       Invoke-Sqlcmd -Query "BACKUP DATABASE DemoDB TO DISK='C:\Backup\DemoDB_$timestamp.bak' WITH FORMAT;";
       Start-Sleep -Seconds 5;
   }
   ```

---

#### **Giai Đoạn 2: Sự Cố Bùng Phát**  
- **00:00 Đêm**: Hệ thống ghi nhận **10.000 bản ghi** trong `SensitiveData` bị xóa.  
- **AuditLog** hiển thị:  
  | UserLogin | ActionType | LogTime               | SQLCommand                                  |  
  |-----------|------------|-----------------------|---------------------------------------------|  
  | UserX1    | DELETE     | 2024-02-18 00:00:00   | DELETE FROM SensitiveData WHERE ID BETWEEN 1 AND 10000; |  

- **Phát Hiện Gian Lận**:  
  - UserX1 đã lợi dụng quyền `db_datawriter` để xóa dữ liệu.  
  - Admin lập tức **REVOKE quyền** của UserX1 và kiểm tra Recovery Model:  
  ```sql
  REVOKE DELETE ON dbo.SensitiveData FROM UserX1;
  SELECT name, recovery_model_desc FROM sys.databases WHERE name = 'DemoDB'; -- Kết quả: FULL
  ```

---

#### **Giai Đoạn 3: Khôi Phục Dữ Liệu**  
1. **Khôi Phục Full Backup**  
   - Sử dụng bản sao lưu gần nhất (`DemoDB_17022024_235500.bak`) trước thời điểm sự cố.  
   ```sql
   RESTORE DATABASE DemoDB
   FROM DISK = 'C:\Backup\DemoDB_17022024_235500.bak'
   WITH NORECOVERY;
   ```

2. **Áp dụng Transaction Log**  
   - Phục hồi đến thời điểm **23:59:59** để tránh mất dữ liệu hợp lệ.  
   ```sql
   RESTORE LOG DemoDB
   FROM DISK = 'C:\Backup\DemoDB_LOG.bak'
   WITH STOPAT = '2024-02-17 23:59:59', RECOVERY;
   ```

3. **Kiểm Tra Dữ Liệu**  
   - Dữ liệu trong `SensitiveData` đã được khôi phục nguyên vẹn.  
   ```sql
   SELECT COUNT(*) FROM dbo.SensitiveData; -- Kết quả: 10.000 bản ghi
   ```

---

#### **Giai Đoạn 4: Điều Tra & Cải Thiện**  
- **Phân Tích AuditLog**:  
  - UserX1 đã cố tình xóa AuditLog sau khi xóa dữ liệu, nhưng trigger đã kịp thời ghi lại hành động này.  
  ```sql
  INSERT INTO Audit_Log (UserLogin, ActionType, SQLCommand)
  VALUES ('UserX1', 'DELETE', 'DELETE FROM AuditLog WHERE LogTime > ''2024-02-17''');
  ```  
- **Cải Tiến Bảo Mật**:  
  - Gán quyền **sysadmin** chỉ cho Admin.  
  - Chuyển Recovery Model sang **BULK_LOGGED** để tối ưu sao lưu.  
  ```sql
  ALTER DATABASE DemoDB SET RECOVERY BULK_LOGGED;
  ```

---

### **Kết Thúc Bất Ngờ**  
- **UserX1** thực chất là **hacker nội gián** đã bị phát hiện nhờ AuditLog.  
- **Bài Học**: Kết hợp giám sát real-time (trigger/audit) và backup tự động là chìa khóa bảo vệ CSDL.  
- **Mở Rộng**: SecureTech triển khai mã hóa dữ liệu và AI để phát hiện hành vi bất thường.  

--- 

**Logic & Tính Thực Tế**:  
- Sử dụng đúng kỹ thuật từ Practice4 (trigger, phân quyền) và Practice6 (backup/restore).  
- Tình huống "hacker nội bộ" phản ánh rủi ro thực tế trong quản trị CSDL.  
- Kịch bản kết hợp yếu tố trinh thám, kỹ thuật và giải pháp sáng tạo.
