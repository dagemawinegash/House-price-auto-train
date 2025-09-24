from django.shortcuts import render
from .ml_model import HousePricePredictor


def predict_price(request):
    prediction = None
    error_message = None

    if request.method == "POST":
        try:
            area = float(request.POST.get("area", 0))
            bedrooms = int(request.POST.get("bedrooms", 0))
            bathrooms = int(request.POST.get("bathrooms", 0))
            stories = int(request.POST.get("stories", 0))
            parking = int(request.POST.get("parking", 0))

            # categorical features
            mainroad = request.POST.get("mainroad", "no")
            guestroom = request.POST.get("guestroom", "no")
            basement = request.POST.get("basement", "no")
            hotwaterheating = request.POST.get("hotwaterheating", "no")
            airconditioning = request.POST.get("airconditioning", "no")
            prefarea = request.POST.get("prefarea", "no")
            furnishingstatus = request.POST.get("furnishingstatus", "unfurnished")

            predictor = HousePricePredictor()
            prediction = predictor.predict(
                area,
                bedrooms,
                bathrooms,
                stories,
                parking,
                mainroad,
                guestroom,
                basement,
                hotwaterheating,
                airconditioning,
                prefarea,
                furnishingstatus,
            )

        except (ValueError, TypeError) as e:
            error_message = f"Invalid input: {str(e)}"
            prediction = None

    return render(
        request,
        "predict.html",
        {"prediction": prediction, "error_message": error_message},
    )
