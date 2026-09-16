"""Export recorded results to a GitHub-readable PDF. Requires reportlab."""
import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, PolyLine, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parent.parent
POLICIES = ['static', 'reactive', 'predictive', 'anticipatory', 'cooperative', 'rl_swarm']
NAMES = ['Static', 'Reactive', 'Predictive', 'Anticipatory', 'Cooperative', 'RL swarm']
COLORS = ['#64748b', '#c65c20', '#a18000', '#00838a', '#6554c0', '#c02b82']


def plot(runs, field, title, factor, max_t, max_y):
    d = Drawing(342, 190)
    d.add(String(38, 174, title, fontName='Helvetica-Bold', fontSize=10))
    for f in (0, .5, 1):
        y = 30 + 126*f
        d.add(Line(38, y, 330, y, strokeColor=colors.HexColor('#dce3ec')))
        d.add(String(31, y-3, f'{max_y*f:.0f}', textAnchor='end', fontSize=8))
    for r in runs:
        points = [v for row in r['series'] for v in (38 + row['tick']/max_t*292, 30 + row[field]/max_y*126)]
        d.add(PolyLine(points, strokeColor=colors.HexColor(COLORS[POLICIES.index(r['policy'])]), strokeWidth=1.3))
    d.add(String(38, 15, '0', fontSize=8))
    d.add(String(330, 15, f'{max_t*factor:.1f} '+('min' if factor != 1 else 'ticks'), textAnchor='end', fontSize=8))
    return d


def build():
    output = ROOT / 'reports' / 'routing-results.pdf'
    output.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles['Normal'].fontSize = 9
    styles['Normal'].leading = 13
    story = []
    sources = [('Synthetic network', ROOT/'results/results.json', 1)] + [
        ('Milpitas - Sunnyvale: '+name, ROOT/f'map_demo/reports/{key}/results.json', 1/6)
        for key, name in [('light', 'Light commute'), ('surge', 'Departure surge'), ('restriction', 'CA 237 capacity stress')]]
    for index, (title, path, factor) in enumerate(sources):
        if index:
            story.append(PageBreak())
        runs = json.loads(path.read_text())['results']
        assert {(r['policy'], r['config']['replan']) for r in runs} == {(p, m) for p in POLICIES for m in (False, True)}
        unit = 'ticks' if factor == 1 else 'min'
        story += [Paragraph('TRAFFIC ROUTING / RESULTS', styles['Heading3']), Paragraph(title, styles['Title']),
                  Paragraph(f"Six policies, two planning modes. {runs[0]['metrics']['vehicles']} simulated vehicles; seed {runs[0]['config']['seed']}. "
                            + ('Time is in abstract ticks.' if factor == 1 else 'Real-road geometry; simulated traffic. One simulator tick is 10 seconds.'), styles['Normal']), Spacer(1, 14)]
        data = [['Policy', 'Planning', f'Mean ({unit})', f'P95 ({unit})', f'Delay\nvehicle-{unit}', 'Peak load', 'Replans', 'Changes']]
        for r in runs:
            m = r['metrics']
            data.append([NAMES[POLICIES.index(r['policy'])], 'Junction' if r['config']['replan'] else 'Departure',
                         f"{m['average_travel_time']*factor:.2f}", f"{m['p95_travel_time']*factor:.2f}",
                         f"{m['total_delay']*factor:.1f}", f"{m['peak_utilization']:.2f}x",
                         str(m['replanning_decisions']), str(m['route_changes'])])
        table = Table(data, colWidths=[100, 80, 85, 80, 105, 80, 70, 70], repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#173448')),
                                   ('TEXTCOLOR', (0,0), (-1,0), colors.white), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                   ('FONTSIZE', (0,0), (-1,-1), 9), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                                   ('ALIGN', (2,0), (-1,-1), 'RIGHT'), ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                                   ('TOPPADDING', (0,0), (-1,-1), 7),
                                   ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4f7')])]))
        story += [table, Spacer(1,12), Paragraph('Replans count junction decisions with multiple legal candidates; changes count decisions that revise the remaining route. '
                   'All trip metrics include background vehicles. Peak load is occupancy divided by capacity times free-flow duration.', styles['Normal']),
                   Spacer(1,6), Paragraph('Limitations: point queues with unlimited storage; no lane or signal model and no live traffic feed. '
                   'RL weights were trained for departure decisions on the corridor network. Junction and synthetic-network results reuse these weights without retraining; these runs are not a new held-out evaluation.', styles['Normal']), PageBreak(),
                   Paragraph(title+' / Traffic over time', styles['Title'])]
        legend = ' &nbsp; '.join(f'<font color="{c}">{n}</font>' for n,c in zip(NAMES,COLORS))
        story += [Paragraph(legend, styles['Normal']), Spacer(1,8)]
        max_t = max(r['series'][-1]['tick'] for r in runs) or 1
        charts = []
        for field, name in [('queued','Vehicles waiting'), ('completed','Completed trips')]:
            max_y = max(row[field] for r in runs for row in r['series']) or 1
            charts.append([plot([r for r in runs if r['config']['replan']==mode], field,
                                name+' / '+('Junction' if mode else 'Departure'), factor, max_t, max_y) for mode in (False, True)])
        story += [Table(charts, colWidths=[345,345]), Spacer(1,8),
                  Paragraph('Both columns use identical axes. Curves may overlap where policies produce the same outcome. '
                            'Values come directly from the recorded simulator runs; the PDF does not rerun or change the experiment.', styles['Normal'])]
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.drawString(36, 18, 'GomathyDhanya/anticipatory-routing | simulated results')
        canvas.drawRightString(756, 18, str(doc.page))
    SimpleDocTemplate(str(output), pagesize=landscape(letter), leftMargin=36, rightMargin=36,
                      topMargin=28, bottomMargin=34, title='Traffic routing - complete results').build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == '__main__':
    build()
