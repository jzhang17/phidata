from django.shortcuts import render
from django.http import JsonResponse
from .models import Research
from .forms import ResearchForm

def index(request):
    if request.method == 'POST':
        form = ResearchForm(request.POST)
        if form.is_valid():
            research = form.save()
            # Here you would typically call your research function
            # and update the research object with the results
            return JsonResponse({'status': 'success', 'id': research.id})
    else:
        form = ResearchForm()
    return render(request, 'research/index.html', {'form': form})

def results(request, research_id):
    research = Research.objects.get(id=research_id)
    return render(request, 'research/results.html', {'research': research})
