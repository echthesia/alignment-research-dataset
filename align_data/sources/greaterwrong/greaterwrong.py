import json
from datetime import datetime
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import requests
from markdownify import markdownify
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
import pytz

from align_data.common.alignment_dataset import AlignmentDataset
from align_data.db.session import make_session
from align_data.db.models import Article, ArticleComment
from align_data.sources.greaterwrong.config import SOURCE_CONFIG, get_source_config
from align_data.settings import LW_GRAPHQL_ACCESS

logger = logging.getLogger(__name__)


class SkipLimitReached(Exception):
    """Raised when the GraphQL API refuses further pagination due to skip limits."""
    pass


def _normalize_timestamp(ts: Optional[datetime]) -> Optional[datetime]:
    if ts and ts.tzinfo is None:
        return ts.replace(tzinfo=pytz.UTC)
    return ts


def get_allowed_tags(name):
    source_config = get_source_config(name)

    if not source_config:
        raise ValueError(
            f'Unknown datasource: "{name}". Must be one of alignmentforum|lesswrong|eaforum'
        )

    # Return the required tags set
    required_tags = set(source_config.get("required_tags", []))
    return required_tags


@dataclass
class GreaterWrong(AlignmentDataset):
    """
    This class allows you to scrape posts and comments from GreaterWrong.
    GreaterWrong contains all the posts from LessWrong (which contains the Alignment Forum) and the EA Forum.
    """

    base_url: str
    start_year: int
    min_karma: int
    """Posts must have at least this much karma to be returned."""
    af: bool
    """Whether alignment forum posts should be returned"""

    limit = 50
    comments_limit = 1000
    recent_comment_post_limit = 50
    recent_comment_pages = None
    COOLDOWN = 0.5
    done_key = "url"
    lazy_eval = True
    source_type = "GreaterWrong"
    _processed_posts: Tuple[Set[str], Set[Tuple[str, str]]] = (set(), set())

    def setup(self):
        super().setup()

        logger.debug("Fetching allowed tags...")
        self.allowed_tags = get_allowed_tags(self.name)
        self._processed_posts = self._load_processed_posts()

    def _comments_table_exists(self, session) -> bool:
        try:
            inspector = inspect(session.bind)
            return inspector.has_table(ArticleComment.__tablename__)
        except Exception as exc:
            logger.error("Failed to inspect database for comments table: %s", exc)
            return False

    def tags_ok(self, post):
        # Check if we should bypass tag checking based on source configuration
        source_config = get_source_config(self.name) or {}
        if source_config.get("bypass_tag_check", False):
            return True

        # Get post tags
        post_tags = {t["name"] for t in post["tags"] if t.get("name")}

        # Check for excluded tags - none must be present
        excluded_tags = set(source_config.get("excluded_tags", []))
        if excluded_tags and excluded_tags.intersection(post_tags):
            return False

        # Check required tags - at least one must be present
        return bool(post_tags & self.allowed_tags)

    def _load_outputted_items(self) -> Set[str]:
        """Return URLs of already processed items (compatible with AlignmentDataset expectations)."""
        return super()._load_outputted_items()

    def _load_processed_posts(self) -> Tuple[Set[str], Set[Tuple[str, str]]]:
        """Load tuples of (url) and (title, authors) for duplicate detection."""
        with make_session() as session:
            articles = (
                session.query(Article.url, Article.title, Article.authors)
                .where(Article.source_type == self.source_type)
                .all()
            )
            return (
                {a.url for a in articles},
                {(a.title.replace("\n", "").strip(), a.authors) for a in articles},
            )

    def not_processed(self, item):
        title = item["title"]
        url = item["pageUrl"]
        authors = ",".join(self.extract_authors(item))

        return (
            url not in self._processed_posts[0]
            and (title, authors) not in self._processed_posts[1]
        )

    @staticmethod
    def _get_published_date(item):
        if isinstance(item, dict):
            item = item.get("postedAt")
        return AlignmentDataset._get_published_date(item)

    def make_query(self, after: str):
        # Get GraphQL query parameters from configuration
        source_config = get_source_config(self.name) or {}
        exclude_events = source_config.get("exclude_events", False)
        karma_threshold = source_config.get("karma_threshold", self.min_karma)

        return f"""
        {{
            posts(input: {{
                terms: {{
                    excludeEvents: {str(exclude_events).lower()}
                    view: "old"
                    af: {json.dumps(self.af)}
                    limit: {self.limit}
                    karmaThreshold: {karma_threshold}
                    after: "{after}"
                    filter: "tagged"
                }}
            }}) {{
                totalCount
                results {{
                    _id
                    title
                    slug
                    pageUrl
                    postedAt
                    modifiedAt
                    score
                    extendedScore
                    baseScore
                    voteCount
                    commentCount
                    wordCount
                    tags {{
                        name
                    }}
                    user {{
                        displayName
                    }}
                    coauthors {{
                        displayName
                    }}
                    af
                    contents {{
                        markdown
                    }}
                    htmlBody
                }}
            }}
        }}
        """

    def _graphql_headers(self) -> Dict[str, str]:
        headers = {
            # The GraphQL endpoint returns a 403 if the user agent isn't set... Makes sense, but is annoying
            "User-Agent": "Mozilla /5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/113.0"
        }

        # Add LessWrong bot-bypass header if configured
        if LW_GRAPHQL_ACCESS:
            header_name, header_value = LW_GRAPHQL_ACCESS.split(":", 1)
            headers[header_name.strip()] = header_value.strip()

        return headers

    def _execute_graphql(self, query: str, result_key: str):
        url = f"{self.base_url}/graphql"
        headers = self._graphql_headers()
        logger.info("Fetching %s from %s", result_key, url)

        try:
            res = requests.post(
                url,
                headers=headers,
                json={"query": query},
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            raise

        logger.info(f"Response status code: {res.status_code}")

        if res.status_code != 200:
            logger.error(f"GraphQL request to {url} failed with status {res.status_code}")
            logger.error(f"Response headers: {dict(res.headers)}")
            logger.error(f"Response body (first 1000 chars): {res.text[:1000]}")
            raise Exception(
                f"GraphQL request to {url} failed with status {res.status_code}. "
                f"Response: {res.text[:200]}"
            )

        try:
            data = res.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response from {url}")
            logger.error(f"Response text (first 1000 chars): {res.text[:1000]}")
            logger.error(f"Parse error: {e}")
            raise Exception(f"Failed to parse JSON from {url}: {e}. Response: {res.text[:200]}")

        if "errors" in data:
            logger.warning("GraphQL errors: %s", data.get("errors"))
            for error in data.get("errors", []):
                if isinstance(error, dict) and "Exceeded maximum value for skip" in str(error.get("message")):
                    logger.warning("Stopping pagination due to skip limit reached")
                    raise SkipLimitReached()

        if "data" not in data:
            logger.error(f"Response missing 'data' field. Response: {data}")
            raise Exception(f"GraphQL response missing 'data' field: {data}")

        if result_key not in data["data"]:
            logger.error(f"Response missing '{result_key}' field. Response: {data}")
            raise Exception(f"GraphQL response missing '{result_key}' field: {data}")

        if data["data"][result_key] is None:
            logger.warning("GraphQL returned null for %s", result_key)
            return {}

        return data["data"][result_key]

    def fetch_posts(self, query: str):
        return self._execute_graphql(query, "posts")

    def make_comments_query(self, post_id: str, offset: int = 0) -> str:
        return f"""
        {{
            comments(input: {{
                terms: {{
                    view: \"postCommentsNew\"
                    postId: \"{post_id}\"
                    limit: {self.comments_limit}
                    offset: {offset}
                }}
            }}) {{
                results {{
                    _id
                    postId
                    parentCommentId
                    postedAt
                    htmlBody
                    contents {{
                        markdown
                    }}
                    pageUrl
                    user {{
                        displayName
                    }}
                }}
            }}
        }}
        """

    def make_recent_comments_query(self, offset: int = 0) -> str:
        return f"""
        {{
            comments(input: {{
                terms: {{
                    view: \"recentComments\"
                    limit: {self.recent_comment_post_limit}
                    offset: {offset}
                }}
            }}) {{
                results {{
                    _id
                    postId
                    post {{
                        _id
                        pageUrl
                        title
                        commentCount
                    }}
                    postedAt
                }}
            }}
        }}
        """

    def fetch_comments_for_post(self, post_id: str) -> List[Dict]:
        comments: List[Dict] = []
        offset = 0

        while True:
            response = self._execute_graphql(
                self.make_comments_query(post_id, offset), "comments"
            )
            if not response or not isinstance(response, dict):
                break

            results = response.get("results", [])
            if not results:
                break

            comments.extend(results)

            if len(results) < self.comments_limit:
                break

            offset += self.comments_limit
            time.sleep(self.COOLDOWN)

        return comments

    def _backfill_comments_by_article_order(self, latest_comment: Optional[datetime]):
        """Fallback path when skip limits are hit: scan recent articles and refresh comments."""
        latest_comment = _normalize_timestamp(latest_comment)
        with make_session() as session:
            articles = (
                session.query(Article)
                .filter(Article.source == self.name)
                .order_by(Article.date_updated.desc().nullslast(), Article.date_published.desc().nullslast())
                .limit(500)
                .all()
            )

            existing_ids = set(
                session.scalars(select(ArticleComment.comment_id)).all()
            )

            new_comments: List[ArticleComment] = []

            for article in articles:
                post_id = article.meta.get("post_id") if isinstance(article.meta, dict) else None
                if not post_id and isinstance(article.meta, str):
                    try:
                        meta_json = json.loads(article.meta)
                        post_id = meta_json.get("post_id")
                    except Exception:
                        pass

                if not post_id:
                    continue

                article_pk = getattr(article, "_id", None)
                if article_pk is None:
                    article_pk = (
                        session.query(Article._id)
                        .filter(Article.url == article.url)
                        .scalar()
                    )
                if article_pk is None:
                    continue

                for comment in self.fetch_comments_for_post(post_id):
                    comment_id = comment.get("_id")
                    if not comment_id or comment_id in existing_ids:
                        continue

                    text = self._comment_text(comment)
                    if not text:
                        continue

                    user = comment.get("user")
                    author = user.get("displayName") if isinstance(user, dict) else None

                    # Create directly with only FK set (no relationship) to avoid SQLAlchemy sync issues
                    comment_model = ArticleComment(
                        comment_id=comment_id,
                        article_id=article_pk,
                        source=self.name,
                        text=text,
                        author=author,
                        posted_at=self._get_published_date(comment.get("postedAt")),
                        parent_comment_id=comment.get("parentCommentId"),
                        url=comment.get("pageUrl"),
                    )

                    comment_ts = _normalize_timestamp(comment_model.posted_at)
                    if latest_comment and comment_ts and comment_ts <= latest_comment:
                        continue

                    new_comments.append(comment_model)
                    existing_ids.add(comment_id)

            if new_comments:
                session.add_all(new_comments)
                session.commit()

    def fetch_recent_comment_page(self, offset: int = 0, before: Optional[str] = None) -> List[Dict]:
        page_results = self._execute_graphql(
            self.make_recent_comments_query(offset), "comments"
        )
        return page_results.get("results", [])

    def _comment_text(self, comment: Dict) -> Optional[str]:
        if comment.get("contents") and comment["contents"].get("markdown"):
            return comment["contents"]["markdown"].strip()

        if comment.get("htmlBody"):
            return markdownify(comment["htmlBody"]).strip()

        return None

    def _comment_to_model(
        self, comment: Dict, article: Article
    ) -> Optional[ArticleComment]:
        """Convert a GraphQL comment response to an ArticleComment model.

        This method handles a SQLAlchemy complexity: when the article is already
        persisted (has a DB id), we set article_id directly to avoid SQLAlchemy
        session warnings about associating objects from different sessions.
        When the article is new (no id yet), we set the relationship instead
        so SQLAlchemy can handle the FK assignment after the article is flushed.
        """
        comment_id = comment.get("_id")
        if not comment_id:
            return None

        text = self._comment_text(comment)
        if not text:
            return None

        user = comment.get("user")
        author = None
        if isinstance(user, dict):
            author = user.get("displayName")

        # If the article already has a DB id, set the FK directly to avoid session warnings.
        article_fk = getattr(article, "_id", None)
        if article_fk is None:
            # Pull identity from SA state if available
            state = getattr(article, "_sa_instance_state", None)
            identity = getattr(state, "identity", None) if state is not None else None
            if identity and len(identity) > 0:
                article_fk = identity[0]

        article_rel = None if article_fk else article

        return ArticleComment(
            comment_id=comment_id,
            article=article_rel,
            article_id=article_fk,
            source=self.name,
            text=text,
            author=author,
            posted_at=self._get_published_date(comment.get("postedAt")),
            parent_comment_id=comment.get("parentCommentId"),
            url=comment.get("pageUrl"),
        )

    def build_article_comments(
        self, article: Article, post: Dict
    ) -> List[ArticleComment]:
        post_id = post.get("_id")
        if not post_id:
            return []

        comments = []
        for comment in self.fetch_comments_for_post(post_id):
            comment_model = self._comment_to_model(comment, article)
            if comment_model:
                comments.append(comment_model)

        return comments

    def update_recent_comments(self):
        page = 0
        offset = 0
        before: Optional[str] = None
        existing_comment_ids: Set[str] = set()
        with make_session() as session:
            if not self._comments_table_exists(session):
                logger.error("Comments table missing; run migrations before fetching comments")
                return

            latest_comment: Optional[datetime] = (
                session.query(ArticleComment.posted_at)
                .filter(ArticleComment.source == self.name)
                .order_by(ArticleComment.posted_at.desc())
                .limit(1)
                .scalar()
            )
            latest_comment = _normalize_timestamp(latest_comment)

        while True:
            try:
                comments = self.fetch_recent_comment_page(offset, before)
            except SkipLimitReached:
                logger.warning("Recent comments skip limit reached; falling back to per-post scan")
                self._backfill_comments_by_article_order(latest_comment)
                break
            except Exception as exc:
                logger.error("Failed to fetch posts with recent comments (offset %s): %s", offset, exc)
                break

            if not comments:
                break

            # comments results embed post info
            posts = [c.get("post") for c in comments if isinstance(c, dict) and c.get("post")]
            url_to_post = {post.get("pageUrl"): post for post in posts if post and post.get("pageUrl")}
            if not url_to_post:
                break

            new_comments_found = False
            processed_any = False
            reached_existing = False

            with make_session() as session:
                # Always consider all known comment ids to avoid uniqueness violations
                existing_comment_ids.update(
                    session.scalars(select(ArticleComment.comment_id)).all()
                )

                articles = (
                    session.query(Article)
                    .filter(Article.source == self.name)
                    .filter(Article.url.in_(url_to_post.keys()))
                    .all()
                )

                if not articles:
                    # None of the recent-comment posts are in our DB; keep paging in case
                    # the next page contains older posts we have stored.
                    pass

                article_ids = [article._id for article in articles if article._id]
                if article_ids:
                    existing_comment_ids.update(
                        session.scalars(
                            select(ArticleComment.comment_id)
                            .where(ArticleComment.article_id.in_(article_ids))
                            .where(ArticleComment.source == self.name)
                        )
                    )

                new_comments: List[ArticleComment] = []

                for article in articles:
                    processed_any = True
                    post_info = url_to_post.get(article.url)
                    if not post_info or not post_info.get("_id"):
                        continue

                    article_pk = getattr(article, "_id", None)
                    if article_pk is None:
                        article_pk = (
                            session.query(Article._id)
                            .filter(Article.url == article.url)
                            .scalar()
                        )
                    if article_pk is None:
                        continue

                    for comment in self.fetch_comments_for_post(post_info["_id"]):
                        comment_id = comment.get("_id")
                        if not comment_id or comment_id in existing_comment_ids:
                            continue

                        text = self._comment_text(comment)
                        if not text:
                            continue

                        user = comment.get("user")
                        author = user.get("displayName") if isinstance(user, dict) else None

                        # Create directly with only FK set (no relationship) to avoid SQLAlchemy sync issues
                        comment_model = ArticleComment(
                            comment_id=comment_id,
                            article_id=article_pk,
                            source=self.name,
                            text=text,
                            author=author,
                            posted_at=self._get_published_date(comment.get("postedAt")),
                            parent_comment_id=comment.get("parentCommentId"),
                            url=comment.get("pageUrl"),
                        )

                        comment_ts = _normalize_timestamp(comment_model.posted_at)
                        if latest_comment and comment_ts and comment_ts <= latest_comment:
                            reached_existing = True
                            break

                        new_comments.append(comment_model)
                        existing_comment_ids.add(comment_id)
                        new_comments_found = True

                    comment_count = post_info.get("commentCount")
                    if comment_count is not None:
                        meta_raw = article.meta
                        meta_dict: Dict[str, Any]
                        if isinstance(meta_raw, str):
                            meta_dict = json.loads(meta_raw) if meta_raw else {}
                        elif isinstance(meta_raw, dict):
                            meta_dict = dict(meta_raw)
                        else:
                            meta_dict = {}

                        if meta_dict.get("comment_count") != comment_count:
                            meta_dict["comment_count"] = comment_count
                            article.meta = meta_dict  # type: ignore[assignment]

                if new_comments:
                    session.add_all(new_comments)

                try:
                    session.commit()
                except IntegrityError as exc:
                    logger.warning("Integrity error writing comments batch: %s", exc)
                    session.rollback()
                    for comment in new_comments:
                        try:
                            session.add(comment)
                            session.commit()
                        except IntegrityError as inner_exc:
                            logger.debug("Skipping duplicate comment %s: %s", comment.comment_id, inner_exc)
                            session.rollback()
                            continue

            if reached_existing:
                break

            if not new_comments_found and processed_any:
                break

            page += 1
            offset = page * self.recent_comment_post_limit

            # Use postedAt of last comment to advance cursor so we avoid skip caps
            last_comment = comments[-1] if comments else None
            if last_comment and last_comment.get("postedAt"):
                before = last_comment.get("postedAt")

            if self.recent_comment_pages and page >= self.recent_comment_pages:
                break

            time.sleep(self.COOLDOWN)

    def fetch_entries(self):
        try:
            self.update_recent_comments()
        except Exception as exc:
            logger.error("Failed to refresh recent comments before fetching entries: %s", exc)

        yield from super().fetch_entries()

    @property
    def last_date_published(self) -> str:
        entries = self.read_entries(sort_by=Article.date_published.desc())
        prev_item = next(iter(entries), None)

        # If there is no previous item or it doesn't have a published date, return default datetime
        if not prev_item or not prev_item.date_published:
            return datetime(self.start_year, 1, 1).isoformat() + "Z"

        # If the previous item has a published date, return it in isoformat
        return prev_item.date_published.isoformat() + "Z"

    @property
    def items_list(self):
        next_date = self.last_date_published
        logger.info("Starting from %s", next_date)
        last_item = None
        while next_date:
            posts = self.fetch_posts(self.make_query(next_date))
            if not posts["results"]:
                return

            # If the only item we find was the one we advanced our iterator to, we're done
            if (
                len(posts["results"]) == 1
                and last_item
                and posts["results"][0]["pageUrl"] == last_item["pageUrl"]
            ):
                return

            for post in posts["results"]:
                # Check if post has content (prefer markdown, fallback to htmlBody)
                has_content = (post.get("contents") and post["contents"].get("markdown")) or post.get("htmlBody")
                if has_content and self.tags_ok(post):
                    yield post

            last_item = posts["results"][-1]
            new_next_date = posts["results"][-1]["postedAt"]
            if next_date == new_next_date:
                raise ValueError(
                    f"could not advance through dataset, next date did not advance after {next_date}"
                )

            next_date = new_next_date
            time.sleep(self.COOLDOWN)

    def extract_authors(self, item):
        authors = item["coauthors"]
        if item["user"]:
            authors = [item["user"]] + authors
        # Some posts don't have authors, for some reaason
        return [a["displayName"] for a in authors] or ["anonymous"]

    def process_entry(self, item):
        # Prefer markdown from contents field (preserves LaTeX), fallback to htmlBody
        if item.get("contents") and item["contents"].get("markdown"):
            text = item["contents"]["markdown"].strip()
        elif item.get("htmlBody"):
            # Fallback to htmlBody (LaTeX will be lost but at least we get the content)
            text = markdownify(item["htmlBody"]).strip()
        else:
            raise ValueError(f"missing both htmlBody and contents.markdown on {item.get('title')!r} from {item.get('url')!r}")

        article = self.make_data_entry(
            {
                "title": item["title"],
                "text": text,
                "url": item["pageUrl"],
                "date_published": self._get_published_date(item),
                "modified_at": item["modifiedAt"],
                "source": self.name,
                "source_type": self.source_type,
                "votes": item["voteCount"],
                "karma": item["baseScore"],
                "tags": [t["name"] for t in item["tags"]],
                "words": item["wordCount"],
                "comment_count": item["commentCount"],
                "authors": self.extract_authors(item),
                "post_id": item.get("_id"),
            }
        )

        article.article_comments = self.build_article_comments(article, item)

        return article
