"""
Reports and Export API routes
"""
from flask import request, jsonify, send_file
from . import api_bp
from db_models import ViolationModel, CameraModel
from datetime import datetime, timedelta
import os
import io
import csv
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile

def get_date_range(date_range_str):
    """Get start and end dates based on date range string"""
    now = datetime.utcnow()
    
    if date_range_str == 'today':
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = now
    elif date_range_str == 'yesterday':
        start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif date_range_str == 'week':
        start_date = now - timedelta(days=7)
        end_date = now
    elif date_range_str == 'month':
        start_date = now - timedelta(days=30)
        end_date = now
    else:
        start_date = now - timedelta(days=30)
        end_date = now
    
    return start_date, end_date

def get_violations_data(date_range_str, report_type='daily'):
    """Fetch violations data based on date range and report type"""
    start_date, end_date = get_date_range(date_range_str)
    
    # Get all violations
    all_violations = ViolationModel.get_all_violations(limit=10000)
    
    # Filter by date
    filtered_violations = []
    for v in all_violations:
        if v.get('created_at'):
            try:
                # Handle different date formats
                created_at_str = v['created_at']
                if isinstance(created_at_str, str):
                    # Try parsing ISO format
                    if 'T' in created_at_str:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        if created_at.tzinfo:
                            created_at = created_at.replace(tzinfo=None)
                    else:
                        created_at = datetime.strptime(created_at_str, '%Y-%m-%d')
                elif isinstance(created_at_str, datetime):
                    created_at = created_at_str
                else:
                    continue
                    
                if start_date <= created_at <= end_date:
                    filtered_violations.append(v)
            except Exception as e:
                # If date parsing fails, skip it for date filtering
                print(f"Date parsing error: {e}, violation: {v.get('_id')}")
                continue
    
    # Get statistics
    stats = ViolationModel.get_violation_stats()
    
    # Get cameras
    cameras = CameraModel.get_all_cameras()
    active_cameras = [c for c in cameras if c.get('status') == 'active']
    
    # Calculate additional stats
    by_type = {}
    by_status = {}
    top_locations = {}
    
    for v in filtered_violations:
        v_type = v.get('type', 'Unknown')
        v_status = v.get('status', 'Unknown')
        v_location = v.get('location', 'Unknown')
        
        by_type[v_type] = by_type.get(v_type, 0) + 1
        by_status[v_status] = by_status.get(v_status, 0) + 1
        top_locations[v_location] = top_locations.get(v_location, 0) + 1
    
    return {
        'violations': filtered_violations,
        'total': len(filtered_violations),
        'by_type': by_type,
        'by_status': by_status,
        'top_locations': dict(sorted(top_locations.items(), key=lambda x: x[1], reverse=True)[:10]),
        'active_cameras': len(active_cameras),
        'start_date': start_date,
        'end_date': end_date
    }

@api_bp.route('/reports/export', methods=['POST'])
def export_report():
    """Export report in specified format (PDF, CSV, Excel)"""
    try:
        data = request.json or {}
        report_type = data.get('report_type', 'daily')
        date_range = data.get('date_range', 'today')
        export_format = data.get('format', 'pdf').lower()
        
        # Get violations data
        report_data = get_violations_data(date_range, report_type)
        
        if export_format == 'pdf':
            return generate_pdf_report(report_data, report_type, date_range)
        elif export_format == 'csv':
            return generate_csv_report(report_data, report_type, date_range)
        elif export_format == 'excel':
            return generate_excel_report(report_data, report_type, date_range)
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported format: {export_format}'
            }), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_pdf_report(report_data, report_type, date_range):
    """Generate PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1E40AF'),
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("Traffic Violation Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report info
    info_style = styles['Normal']
    story.append(Paragraph(f"<b>Report Type:</b> {report_type.title()}", info_style))
    story.append(Paragraph(f"<b>Date Range:</b> {date_range.title()}", info_style))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", info_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary statistics
    story.append(Paragraph("<b>Summary Statistics</b>", styles['Heading2']))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Violations', str(report_data['total'])],
        ['Active Cameras', str(report_data['active_cameras'])],
        ['Date Range', f"{report_data['start_date'].strftime('%Y-%m-%d')} to {report_data['end_date'].strftime('%Y-%m-%d')}"]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Violations by type
    if report_data['by_type']:
        story.append(Paragraph("<b>Violations by Type</b>", styles['Heading2']))
        type_data = [['Type', 'Count']]
        for v_type, count in sorted(report_data['by_type'].items(), key=lambda x: x[1], reverse=True):
            type_data.append([v_type, str(count)])
        
        type_table = Table(type_data, colWidths=[3*inch, 2*inch])
        type_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#059669')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(type_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Top locations
    if report_data['top_locations']:
        story.append(Paragraph("<b>Top Violation Locations</b>", styles['Heading2']))
        location_data = [['Location', 'Violations']]
        for location, count in list(report_data['top_locations'].items())[:10]:
            location_data.append([location, str(count)])
        
        location_table = Table(location_data, colWidths=[4*inch, 1*inch])
        location_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#DC2626')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(location_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Detailed violations (first 50)
    if report_data['violations']:
        story.append(PageBreak())
        story.append(Paragraph("<b>Detailed Violations</b>", styles['Heading2']))
        
        violations_data = [['ID', 'Type', 'Status', 'Location', 'Date', 'Confidence']]
        for v in report_data['violations'][:50]:  # Limit to 50 for PDF
            violations_data.append([
                str(v.get('_id', ''))[:8],
                v.get('type', 'Unknown'),
                v.get('status', 'Unknown'),
                v.get('location', 'Unknown')[:30],
                v.get('created_at', '')[:10] if v.get('created_at') else '',
                f"{v.get('confidence', 0)*100:.1f}%" if v.get('confidence') else 'N/A'
            ])
        
        violations_table = Table(violations_data, colWidths=[1*inch, 1*inch, 1*inch, 2*inch, 1*inch, 1*inch])
        violations_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7C3AED')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(violations_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    filename = f"violation_report_{report_type}_{date_range}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )

def generate_csv_report(report_data, report_type, date_range):
    """Generate CSV report"""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    
    # Write header
    writer.writerow(['Traffic Violation Report'])
    writer.writerow(['Report Type', report_type])
    writer.writerow(['Date Range', date_range])
    writer.writerow(['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    writer.writerow([])
    
    # Summary
    writer.writerow(['Summary Statistics'])
    writer.writerow(['Total Violations', report_data['total']])
    writer.writerow(['Active Cameras', report_data['active_cameras']])
    writer.writerow([])
    
    # Violations by type
    if report_data['by_type']:
        writer.writerow(['Violations by Type'])
        writer.writerow(['Type', 'Count'])
        for v_type, count in sorted(report_data['by_type'].items(), key=lambda x: x[1], reverse=True):
            writer.writerow([v_type, count])
        writer.writerow([])
    
    # Detailed violations
    writer.writerow(['Detailed Violations'])
    writer.writerow(['ID', 'Type', 'Status', 'Location', 'Confidence', 'Frame', 'Timestamp', 'Created At'])
    for v in report_data['violations']:
        writer.writerow([
            str(v.get('_id', '')),
            v.get('type', ''),
            v.get('status', ''),
            v.get('location', ''),
            f"{v.get('confidence', 0)*100:.2f}%" if v.get('confidence') else '',
            v.get('frame', ''),
            v.get('timestamp', ''),
            v.get('created_at', '')
        ])
    
    # Convert to bytes
    csv_bytes = io.BytesIO()
    csv_bytes.write(buffer.getvalue().encode('utf-8'))
    csv_bytes.seek(0)
    
    filename = f"violation_report_{report_type}_{date_range}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(
        csv_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

def generate_excel_report(report_data, report_type, date_range):
    """Generate Excel report"""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Summary sheet
        summary_df = pd.DataFrame({
            'Metric': ['Total Violations', 'Active Cameras', 'Date Range'],
            'Value': [
                report_data['total'],
                report_data['active_cameras'],
                f"{report_data['start_date'].strftime('%Y-%m-%d')} to {report_data['end_date'].strftime('%Y-%m-%d')}"
            ]
        })
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Violations by type
        if report_data['by_type']:
            type_df = pd.DataFrame(list(report_data['by_type'].items()), columns=['Type', 'Count'])
            type_df = type_df.sort_values('Count', ascending=False)
            type_df.to_excel(writer, sheet_name='By Type', index=False)
        
        # Violations by status
        if report_data['by_status']:
            status_df = pd.DataFrame(list(report_data['by_status'].items()), columns=['Status', 'Count'])
            status_df = status_df.sort_values('Count', ascending=False)
            status_df.to_excel(writer, sheet_name='By Status', index=False)
        
        # Top locations
        if report_data['top_locations']:
            location_df = pd.DataFrame(list(report_data['top_locations'].items()), columns=['Location', 'Violations'])
            location_df = location_df.sort_values('Violations', ascending=False)
            location_df.to_excel(writer, sheet_name='Top Locations', index=False)
        
        # Detailed violations
        if report_data['violations']:
            violations_list = []
            for v in report_data['violations']:
                violations_list.append({
                    'ID': str(v.get('_id', '')),
                    'Type': v.get('type', ''),
                    'Status': v.get('status', ''),
                    'Location': v.get('location', ''),
                    'Confidence': f"{v.get('confidence', 0)*100:.2f}%" if v.get('confidence') else '',
                    'Frame': v.get('frame', ''),
                    'Timestamp': v.get('timestamp', ''),
                    'Created At': v.get('created_at', ''),
                    'Description': v.get('description', '')
                })
            violations_df = pd.DataFrame(violations_list)
            violations_df.to_excel(writer, sheet_name='Violations', index=False)
    
    buffer.seek(0)
    
    filename = f"violation_report_{report_type}_{date_range}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )

