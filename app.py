import os
from functools import wraps

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from config import AppConfig
from myutils.admin_api import admin_bp
from myutils.app_logging import setup_logging
from myutils.behaviorData import add_behavior, get_behavior_snapshot, get_user_behavior
from myutils.query import (
    authenticate_user,
    create_user,
    init_db,
)
from services.catalog_service import CatalogService
from services.recommendation_service import RecommendationService


app = Flask(__name__)
app.config.from_object(AppConfig)
setup_logging(app)
init_db()
app.register_blueprint(admin_bp)
catalog_service = CatalogService()
recommendation_service = RecommendationService()


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("email"):
            if request.path.startswith("/behavior"):
                return jsonify({"code": 401, "msg": "请先登录后再操作"}), 401
            flash("请先登录后再访问该页面。", "warning")
            return redirect(url_for("login", next=request.url))
        return view_func(*args, **kwargs)

    return wrapped_view


@app.context_processor
def inject_user_context():
    return {
        "current_user_email": session.get("email"),
        "current_username": session.get("username"),
        "is_admin_user": bool(session.get("email")) and session.get("email") in app.config["ADMIN_EMAILS"],
    }


@app.route("/")
def index():
    top_movies = catalog_service.top_movies(10)
    recommend_list = recommendation_service.recommend_for_user(session.get("email"), 6) if session.get("email") else []
    return render_template(
        "index.html",
        top_movies=top_movies,
        recommend_list=recommend_list,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("请完整填写注册信息。", "danger")
            return render_template("register.html")
        if password != confirm_password:
            flash("两次输入的密码不一致。", "danger")
            return render_template("register.html")

        success, message = create_user(username=username, email=email, password=password)
        flash(message, "success" if success else "danger")
        if success:
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", request.form.get("username", "")).strip().lower()
        password = request.form.get("password", "")
        user = authenticate_user(email, password)
        if user:
            session["email"] = user["email"]
            session["username"] = user["username"]
            flash("登录成功。", "success")
            next_url = request.args.get("next")
            return redirect(next_url or url_for("index"))
        flash("邮箱或密码错误。", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("已退出登录。", "success")
    return redirect(url_for("index"))


@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    movies = catalog_service.search_movies(query) if query else []
    return render_template("search.html", query=query, movies=movies)


@app.route("/movie/<int:movie_id>")
def movie(movie_id):
    movie_data = catalog_service.movie_detail(movie_id)
    if not movie_data:
        abort(404)

    comments = catalog_service.movie_comments(movie_id)
    recommendations = recommendation_service.recommend_similar(movie_id, 6)
    user_behaviors = (
        get_behavior_snapshot(session.get("email"), movie_id) if session.get("email") else {}
    )
    return render_template(
        "movie.html",
        movie=movie_data,
        recommendations=recommendations,
        comments=comments,
        user_behaviors=user_behaviors,
    )


@app.route("/behavior/add", methods=["POST"])
@login_required
def add_user_behavior():
    payload = request.get_json(silent=True) or request.form
    try:
        movie_id = int(payload.get("movie_id", 0))
        behavior_type = int(payload.get("behavior_type", 0))
        score_raw = payload.get("score")
        score = float(score_raw) if score_raw not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"code": 400, "msg": "请求参数格式错误"}), 400

    if movie_id <= 0 or behavior_type not in {1, 2, 3}:
        return jsonify({"code": 400, "msg": "无效的行为参数"}), 400

    try:
        add_behavior(session["email"], movie_id, behavior_type, score)
    except ValueError as exc:
        app.logger.warning("behavior rejected: %s", exc)
        return jsonify({"code": 400, "msg": str(exc)}), 400

    app.logger.info(
        "behavior accepted | user=%s movie_id=%s behavior_type=%s score=%s",
        session["email"],
        movie_id,
        behavior_type,
        score,
    )
    return jsonify({"code": 200, "msg": "行为提交成功"})


@app.route("/behavior/get")
@login_required
def get_behavior():
    behavior_list = get_user_behavior(session["email"])
    return jsonify({"code": 200, "data": behavior_list})


@app.route("/recommend")
@login_required
def get_recommend():
    top_n = request.args.get("top_n", default=10, type=int)
    recommend_list = recommendation_service.recommend_for_user(session["email"], top_n)
    return render_template(
        "recommend.html",
        email=session["email"],
        recommend_list=recommend_list,
        top_n=top_n,
    )


@app.errorhandler(401)
def handle_unauthorized(_error):
    if request.path.startswith("/api/"):
        return jsonify({"code": 401, "msg": "请先登录"}), 401
    flash("请先登录后再访问后台页面。", "warning")
    return redirect(url_for("login", next=request.url))


@app.errorhandler(403)
def handle_forbidden(_error):
    if request.path.startswith("/api/"):
        return jsonify({"code": 403, "msg": "没有管理员权限"}), 403
    flash("当前账号没有管理员权限。", "danger")
    return redirect(url_for("index"))


@app.errorhandler(404)
def handle_not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"code": 404, "msg": "资源不存在"}), 404
    flash("页面不存在。", "warning")
    return redirect(url_for("index"))


@app.errorhandler(500)
def handle_server_error(error):
    app.logger.exception("unhandled server error: %s", error)
    if request.path.startswith("/api/"):
        return jsonify({"code": 500, "msg": "服务器内部错误"}), 500
    flash("系统开小差了，请稍后重试。", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "1").strip().lower() not in {"0", "false", "no"}
    host = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("PORT", "5001"))
    app.run(host=host, port=port, debug=debug)
