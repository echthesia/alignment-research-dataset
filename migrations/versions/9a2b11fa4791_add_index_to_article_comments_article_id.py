"""add index to article_comments article_id

Revision ID: 9a2b11fa4791
Revises: d1bfe8f5c4e7
Create Date: 2025-11-24 17:27:21.009319

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9a2b11fa4791'
down_revision = 'd1bfe8f5c4e7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_article_comments_article_id", "article_comments", ["article_id"])


def downgrade() -> None:
    op.drop_index("ix_article_comments_article_id", table_name="article_comments")
