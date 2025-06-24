function predict_prices() {
    const area = parseFloat(document.getElementById("area").value);
    const total_rooms = parseFloat(document.getElementById("rooms").value);
    const stories = parseFloat(document.getElementById("stories").value);
    const parkingValue = document.getElementById("parking").value;
    const mainroadValue = document.getElementById("mainroad").value;
    const has_parking = parkingValue === "YES" ? 1 : 0;
    const mainroad = mainroadValue === "YES" ? 1 : 0;

    if (area <= 0 || total_rooms <= 0) {
        document.getElementById("result").innerText = "Area and total rooms must be > 0";
        return;
    }

    const log_area = Math.log(area);
    const log_price = 
        11.4929 +
        0.3587 * log_area +
        0.1020 * total_rooms +
        0.0940 * stories +
        0.0685 * has_parking +
        0.1474 * mainroad;

    const price = Math.exp(log_price);

    document.getElementById("result").innerText = `Predicted Price: ₹ ${price.toFixed(2)}`;
}
