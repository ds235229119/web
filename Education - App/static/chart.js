function getPrediction() {
    let data = {
        hours_studied: parseFloat(document.getElementById('hours_studied').value),
        previous_grades: parseFloat(document.getElementById('previous_grades').value),
        attendance: parseFloat(document.getElementById('attendance').value)
    };

    // Check if any field is empty or invalid
    if (Object.values(data).some(value => isNaN(value))) {
        document.getElementById('result').innerText = "❌ Invalid input! Please enter valid values.";
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerText = `⚠️ Error: ${data.error}`;
        } else {
            document.getElementById('result').innerText = `🎯 Predicted Final Grade: ${data.final_grade_prediction}`;
            updateChart(data.final_grade_prediction);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = "⚠️ Server error. Please try again later.";
    });
}

function updateChart(predictedGrade) {
    let ctx = document.getElementById('performanceChart').getContext('2d');

    // Destroy previous chart instance to avoid duplicates
    if (window.myChart) {
        window.myChart.destroy();
    }

    window.myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Predicted Grade'],
            datasets: [{
                label: 'Grade Prediction',
                data: [predictedGrade],
                backgroundColor: 'rgba(54, 162, 235, 0.7)'
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}
